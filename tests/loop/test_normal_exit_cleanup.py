"""Normal worker exit reaps retained descendants before stream EOF."""

from __future__ import annotations

import os
import signal
import sys
import threading
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
    _ResolvedAgentRun,  # pyright: ignore[reportPrivateUsage]
    _run_agent_streaming,  # pyright: ignore[reportPrivateUsage]
)
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions
from milknado.loop._process_identity import identity_state
from milknado.loop._process_lifecycle import ProtectedWorker, spawn_protected

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX identity cleanup is required")

_PROGRAM = """
# BEGIN WORKER
import json
import subprocess
import sys
import time
from pathlib import Path

marker, release = map(Path, sys.argv[1:3])
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    stdout=sys.stdout,
    stderr=sys.stderr,
    start_new_session=True,
)
marker.write_text(str(child.pid))
while not release.exists():
    time.sleep(0.02)
print(json.dumps({"type": "result", "result": "leader-output"}), flush=True)
print("leader-error", file=sys.stderr, flush=True)
raise SystemExit(7)
# END WORKER
"""


def _until(predicate: Callable[[], bool], timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _observed_child(
    graph: MikadoGraph, worker: ProtectedWorker, marker: Path
) -> tuple[int, float]:
    assert _until(marker.exists)
    child_pid = int(marker.read_text())
    assert _until(
        lambda: (
            (record := graph.runs.get_worker(worker.identity.invocation_id)) is not None
            and any(pid == child_pid for pid, _token, _pgid in record.descendants)
        )
    )
    token = psutil.Process(child_pid).create_time()
    assert identity_state(child_pid, token) == "live"
    return child_pid, token


def _assert_finished(
    graph: MikadoGraph, worker: ProtectedWorker, result: AgentResult, child: tuple[int, float]
) -> None:
    assert result.returncode == 7
    assert result.result_text == "leader-output"
    assert result.captured_stdout is not None
    assert "leader-output" in result.captured_stdout
    assert result.captured_stderr is not None
    assert "leader-error" in result.captured_stderr
    assert worker.finish(timeout=3)
    record = graph.runs.get_worker(worker.identity.invocation_id)
    assert record is not None and record.ended_at is not None
    assert record.ready_generation == 0
    assert identity_state(*child) == "gone"


def _overlap_shutdown(
    worker: ProtectedWorker, release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    proceed = threading.Event()
    original = worker._normal_exit  # pyright: ignore[reportPrivateUsage]

    def observed_exit() -> None:
        entered.set()
        assert proceed.wait(3)
        original()

    monkeypatch.setattr(worker, "_normal_exit", observed_exit)
    release.touch()
    assert worker.process.wait(timeout=3) == 7
    assert entered.wait(3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stopped = pool.submit(worker.shutdown, time.monotonic() + 4)
        proceed.set()
        assert stopped.result(timeout=5)


def _kill_helper(graph: MikadoGraph, worker: ProtectedWorker) -> None:
    record = graph.runs.get_worker(worker.identity.invocation_id)
    assert record is not None and record.helper_pid is not None
    os.kill(record.helper_pid, signal.SIGKILL)


@pytest.mark.parametrize(
    ("helper_dead", "overlap"),
    [(False, False), (True, False), (False, True)],
    ids=["helper-alive", "helper-dead", "concurrent-shutdown"],
)
def test_normal_exit_reaps_retained_child_without_caller_cleanup(
    tmp_path: Path, helper_dead: bool, overlap: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    supervisor = psutil.Process()
    context = ProtectionContext(
        LoopWorkerEvidence(graph.db_path),
        WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
        graph.db_path,
    )
    workers: list[ProtectedWorker] = []
    marker, release = tmp_path / "child.pid", tmp_path / "release"

    def launch(options: SpawnOptions) -> ProtectedWorker:
        worker = spawn_protected(options, context)
        workers.append(worker)
        return worker

    run = _ResolvedAgentRun(
        [sys.executable, "-c", _PROGRAM, str(marker), str(release)],
        None,
        timeout=10,
        log_dir=tmp_path,
        iteration=1,
        spawn_worker=launch,
        capture_result_text=True,
    )
    child_pid: int | None = None
    child_token: float | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result_future = pool.submit(_run_agent_streaming, run)
            try:
                assert _until(lambda: len(workers) == 1)
                worker = workers[0]
                child_pid, child_token = _observed_child(graph, worker, marker)
                if overlap:
                    _overlap_shutdown(worker, release, monkeypatch)
                else:
                    release.touch()
                    assert worker.process.wait(timeout=3) == 7
                if helper_dead:
                    _kill_helper(graph, worker)
                result = result_future.result(timeout=5)
                _assert_finished(graph, worker, result, (child_pid, child_token))
            finally:
                release.touch()
                if (
                    child_pid is not None
                    and child_token is not None
                    and identity_state(child_pid, child_token) == "live"
                ):
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    finally:
        graph.close()
