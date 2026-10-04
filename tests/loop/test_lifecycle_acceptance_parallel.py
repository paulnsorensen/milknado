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
    _ResolvedAgentRun,  # pyright: ignore[reportPrivateUsage]
    _run_agent_blocking,  # pyright: ignore[reportPrivateUsage]
    _run_agent_streaming,  # pyright: ignore[reportPrivateUsage]
)
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions
from milknado.loop._process_lifecycle import ProtectedWorker, spawn_protected

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX helper uses passed file descriptors"
)

_PROGRAM = """
import os
import sys
import time
from pathlib import Path
stage, release = map(Path, sys.argv[1:3])
stage.write_text(str(os.getpid()))
print('started', flush=True)
while not release.exists():
    time.sleep(.02)
print('{"type":"result","result":"' + sys.argv[3] + '"}', flush=True)
raise SystemExit(int(sys.argv[4]))
"""


def _until(predicate: Callable[[], bool], timeout: float = 8) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _assert_isolated_takeover(
    graph: MikadoGraph, affected: ProtectedWorker, other: ProtectedWorker, tmp_path: Path
) -> None:
    first_record = graph.runs.get_worker(affected.identity.invocation_id)
    other_record = graph.runs.get_worker(other.identity.invocation_id)
    assert first_record is not None and first_record.helper_pid is not None
    assert other_record is not None and other_record.helper_pid is not None
    assert (tmp_path / "affected").read_text() == str(first_record.pid)
    assert (tmp_path / "other").read_text() == str(other_record.pid)
    os.kill(first_record.helper_pid, signal.SIGKILL)
    assert _until(
        lambda: (
            (record := graph.runs.get_worker(affected.identity.invocation_id)) is not None
            and record.ready_generation == 1
        )
    )
    same_other = graph.runs.get_worker(other.identity.invocation_id)
    assert same_other is not None and same_other.helper_pid == other_record.helper_pid
    assert same_other.ready_generation == 0
    assert other.process.poll() is None


@pytest.mark.parametrize("streaming", [False, True], ids=["blocking", "streaming"])
def test_successful_replacement_does_not_interrupt_concurrent_generic_run(
    tmp_path: Path,
    streaming: bool,
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    supervisor = psutil.Process()
    owner_pid, owner_token = supervisor.pid, supervisor.create_time()
    workers: dict[str, ProtectedWorker] = {}
    releases = [tmp_path / "release-a", tmp_path / "release-b"]

    def start(label: str, code: int) -> _ResolvedAgentRun:
        node = graph.add_node(label)
        graph.runs.start(label, node.id, f"{label}.log", "2026-01-01T00:00:00+00:00", None)
        context = ProtectionContext(
            LoopWorkerEvidence(graph.db_path),
            WorkerOwner(label, owner_pid, owner_token, label, node.id),
            graph.db_path,
        )

        def launch(options: SpawnOptions) -> ProtectedWorker:
            worker = spawn_protected(options, context)
            workers[label] = worker
            return worker

        index = 0 if label == "affected" else 1
        return _ResolvedAgentRun(
            [
                sys.executable,
                "-c",
                _PROGRAM,
                str(tmp_path / label),
                str(releases[index]),
                label,
                str(code),
            ],
            None,
            timeout=10,
            log_dir=tmp_path,
            iteration=index + 1,
            spawn_worker=launch,
            capture_result_text=True,
        )

    execute = _run_agent_streaming if streaming else _run_agent_blocking
    try:
        affected = start("affected", 17)
        other = start("other", 19)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(execute, affected)
            second = pool.submit(execute, other)
            assert _until(lambda: (tmp_path / "affected").exists())
            assert _until(lambda: (tmp_path / "other").exists())
            first_worker = workers["affected"]
            other_worker = workers["other"]
            _assert_isolated_takeover(graph, first_worker, other_worker, tmp_path)
            releases[1].touch()
            releases[0].touch()
            first_result, other_result = first.result(timeout=10), second.result(timeout=10)
        assert (first_result.returncode, first_result.result_text) == (17, "affected")
        assert (other_result.returncode, other_result.result_text) == (19, "other")
        assert graph.runs.live_workers(run_id="affected") == ()
        assert graph.runs.live_workers(run_id="other") == ()
    finally:
        for release in releases:
            release.touch()
        for worker in workers.values():
            if worker.process.poll() is None:
                _ = worker.shutdown(time.monotonic() + 3)
        graph.close()
