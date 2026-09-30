from __future__ import annotations

import os
import signal
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._agent import (
    AgentResult,
    AgentRunSpec,
    _ResolvedAgentRun,  # pyright: ignore[reportPrivateUsage]
    _run_agent_blocking,  # pyright: ignore[reportPrivateUsage]
    _run_agent_streaming,  # pyright: ignore[reportPrivateUsage]
)
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions
from milknado.loop._process_lifecycle import ProtectedWorker, spawn_protected
from milknado.loop.sessions import SessionChannel, run_session

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX helper uses passed file descriptors"
)

_GENERIC = """
import os
import sys
import time
from pathlib import Path
stage, release = map(Path, sys.argv[1:3])
stage.write_text(str(os.getpid()))
print('started', flush=True)
while not release.exists():
    time.sleep(.02)
print('{"type":"result","result":"finished"}', flush=True)
raise SystemExit(19)
"""

_NATIVE = """
import json
import os
import sys
import time
from pathlib import Path
stage, release = map(Path, sys.argv[1:3])
for _ in range(2):
    if not sys.stdin.readline():
        raise SystemExit(2)
stage.write_text(str(os.getpid()))
print(json.dumps({'type':'assistant','session_id':'sid','message':
    {'role':'assistant','content':[{'type':'text','text':'started'}]}}), flush=True)
while not release.exists():
    time.sleep(.02)
print(json.dumps({'type':'result','subtype':'success','result':'finished',
                  'session_id':'sid'}), flush=True)
raise SystemExit(19)
"""


def _until(predicate: Callable[[], bool], timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _start_run(graph: MikadoGraph, label: str) -> ProtectionContext:
    node = graph.add_node(label)
    graph.runs.start(label, node.id, f"{label}.log", "2026-01-01T00:00:00+00:00", None)
    supervisor = psutil.Process()
    owner = WorkerOwner(label, supervisor.pid, supervisor.create_time(), label, node.id)
    return ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path)


def _launch(
    context: ProtectionContext, workers: dict[str, ProtectedWorker], label: str
) -> Callable[[SpawnOptions], ProtectedWorker]:
    def spawn(options: SpawnOptions) -> ProtectedWorker:
        worker = spawn_protected(options, context)
        workers[label] = worker
        return worker

    return spawn


def _generic_run(
    tmp_path: Path, label: str, spawn: Callable[[SpawnOptions], ProtectedWorker], iteration: int
) -> _ResolvedAgentRun:
    return _ResolvedAgentRun(
        [
            sys.executable,
            "-c",
            _GENERIC,
            str(tmp_path / label),
            str(tmp_path / f"release-{label}"),
        ],
        None,
        timeout=20,
        log_dir=tmp_path,
        iteration=iteration,
        spawn_worker=spawn,
        capture_result_text=True,
    )


def _native_run(
    tmp_path: Path, stage: Path, release: Path, spawn: Callable[[SpawnOptions], ProtectedWorker]
) -> AgentRunSpec:
    script = tmp_path / "native.py"
    _ = script.write_text(_NATIVE)
    executable = tmp_path / "claude"
    executable.symlink_to(sys.executable)
    return AgentRunSpec(
        cmd=[str(executable), str(script), str(stage), str(release)],
        prompt="request",
        timeout=20,
        log_dir=None,
        iteration=1,
        capture_result_text=True,
        cwd=tmp_path,
        spawn_worker=spawn,
    )


def _ready(graph: MikadoGraph, worker: ProtectedWorker, generation: int) -> bool:
    record = graph.runs.get_worker(worker.identity.invocation_id)
    return record is not None and record.ready_generation == generation


def _exhaust(
    graph: MikadoGraph,
    worker: ProtectedWorker,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if failure == "hung":
        fake = graph.db_path.parent / "hung-helper"
        _ = fake.write_text("#!/bin/sh\nexec sleep 30\n")
        fake.chmod(0o700)
        monkeypatch.setattr(sys, "executable", str(fake))
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.helper_pid is not None
        os.kill(record.helper_pid, signal.SIGKILL)
        assert _until(lambda: worker.process.poll() is not None)
        return

    for generation in range(4):
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.helper_pid is not None
        assert record.ready_generation == generation
        os.kill(record.helper_pid, signal.SIGKILL)
        if generation < 3:
            assert _until(lambda expected=generation + 1: _ready(graph, worker, expected))
            assert worker.process.poll() is None
    assert _until(lambda: worker.process.poll() is not None)


def _assert_started(
    graph: MikadoGraph, workers: dict[str, ProtectedWorker], tmp_path: Path
) -> int:
    affected = graph.runs.get_worker(workers["affected"].identity.invocation_id)
    other = graph.runs.get_worker(workers["other"].identity.invocation_id)
    assert affected is not None and affected.helper_pid is not None
    assert other is not None and other.helper_pid is not None
    assert (tmp_path / "affected").read_text() == str(affected.pid)
    assert (tmp_path / "other").read_text() == str(other.pid)
    return other.helper_pid


def _assert_failure_and_isolation(
    graph: MikadoGraph,
    workers: dict[str, ProtectedWorker],
    failed: AgentResult,
    other_pid: int,
) -> None:
    affected = graph.runs.get_worker(workers["affected"].identity.invocation_id)
    other = graph.runs.get_worker(workers["other"].identity.invocation_id)
    assert affected is not None and affected.ended_at is not None
    assert graph.runs.live_workers(run_id="affected") == ()
    assert failed.returncode is not None and failed.returncode != 0
    assert failed.result_text != "finished"
    assert other is not None and other.helper_pid == other_pid
    assert other.ready_generation == 0 and workers["other"].process.poll() is None


@pytest.mark.parametrize("flavor", ["blocking", "streaming", "native"])
@pytest.mark.parametrize("failure", ["crash", "hung"])
def test_agent_exhaustion_closes_only_affected_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flavor: str,
    failure: str,
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    workers: dict[str, ProtectedWorker] = {}
    affected_stage, other_stage = tmp_path / "affected", tmp_path / "other"
    affected_release, other_release = tmp_path / "release-affected", tmp_path / "release-other"
    affected_spawn = _launch(_start_run(graph, "affected"), workers, "affected")
    other_spawn = _launch(_start_run(graph, "other"), workers, "other")
    affected = (
        _native_run(tmp_path, affected_stage, affected_release, affected_spawn)
        if flavor == "native"
        else _generic_run(tmp_path, "affected", affected_spawn, 1)
    )
    other = _generic_run(tmp_path, "other", other_spawn, 2)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            if flavor == "native":
                assert isinstance(affected, AgentRunSpec)
                first = pool.submit(run_session, affected, SessionChannel())
            else:
                assert isinstance(affected, _ResolvedAgentRun)
                execute = _run_agent_streaming if flavor == "streaming" else _run_agent_blocking
                first = pool.submit(execute, affected)
            second = pool.submit(_run_agent_blocking, other)
            try:
                assert _until(affected_stage.exists)
                assert _until(other_stage.exists)
                affected_worker = workers["affected"]
                other_pid = _assert_started(graph, workers, tmp_path)
                _exhaust(graph, affected_worker, failure, monkeypatch)
                failed = first.result(timeout=12)
                _assert_failure_and_isolation(graph, workers, failed, other_pid)
                other_release.touch()
                succeeded = second.result(timeout=5)
            finally:
                affected_release.touch()
                other_release.touch()
        assert (succeeded.returncode, succeeded.result_text) == (19, "finished")
        assert graph.runs.live_workers(run_id="other") == ()
    finally:
        for worker in workers.values():
            if worker.process.poll() is None:
                _ = worker.shutdown(time.monotonic() + 3)
        graph.close()
