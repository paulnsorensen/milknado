from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Timer

import psutil
import pytest
from typing_extensions import override

from milknado.adapters.process import ProcessAdapter
from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.dispatch import ReapRequest, reap_orphaned_workers
from milknado.domains.graph import MikadoGraph, NodeWorkers

if os.name != "posix":
    pytest.skip("POSIX process identities are required", allow_module_level=True)


@pytest.fixture
def worker() -> Iterator[psutil.Process]:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield psutil.Process(child.pid)
    finally:
        child.kill()
        _ = child.wait(timeout=5)


def _recorded_graph(tmp_path: Path, worker: psutil.Process, helper: HelperIdentity) -> MikadoGraph:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("recover")
    assert graph.claim_node(node.id, "run-1", now="2026-01-01T00:00:00+00:00", pid=2**31 - 1)
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    graph.runs.record_worker(
        WorkerOwner("run-1", 2**31 - 1, 123.5, "run-1", node.id),
        WorkerIdentity("inv-1", worker.pid, worker.pid, worker.create_time()),
    )
    graph.runs.record_helper(helper)
    return graph


def _observing_graph(
    tmp_path: Path, worker: psutil.Process, helper: HelperIdentity
) -> MikadoGraph:
    graph = _recorded_graph(tmp_path, worker, helper)
    graph.runs.begin_worker_observation(_helper_key(helper))
    return graph


def _reap(graph: MikadoGraph, step: Callable[[], None]) -> bool:
    pending_step: Callable[[], None] | None = step

    class _HelperRace(ProcessAdapter):
        """Run one helper step after reap reads the worker record and before it begins."""

        @staticmethod
        @override
        def supervisor_state(pid: int, start_token: float) -> str:
            nonlocal pending_step
            if pending_step is not None:
                callback, pending_step = pending_step, None
                callback()
            return ProcessAdapter.supervisor_state(pid, start_token)

    request = ReapRequest(NodeWorkers(1), deadline=time.monotonic() + 5)
    return reap_orphaned_workers(graph, _HelperRace(), request)


def _helper_key(helper: HelperIdentity) -> ObservationKey:
    return ObservationKey("inv-1", "helper", 1, helper.generation, helper.pid, helper.start_token)


def test_reap_waits_for_live_helper_observation(tmp_path: Path, worker: psutil.Process) -> None:
    current = psutil.Process()
    helper = HelperIdentity("inv-1", 0, current.pid, current.create_time())
    graph = _observing_graph(tmp_path, worker, helper)
    commit = Timer(0.3, graph.runs.commit_worker_observation, (_helper_key(helper), ()))
    try:
        commit.start()
        assert reap_orphaned_workers(
            graph, ProcessAdapter(), ReapRequest(NodeWorkers(1), deadline=time.monotonic() + 5)
        )
        assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE
        recorded = graph.runs.get_worker("inv-1")
        assert recorded is not None and recorded.ended_at is not None
    finally:
        commit.join()
        graph.close()


def test_reap_preserves_worker_when_helper_died_mid_observation(
    tmp_path: Path, worker: psutil.Process, caplog: pytest.LogCaptureFixture
) -> None:
    helper = HelperIdentity("inv-1", 0, 2**31 - 1, 123.5)
    graph = _observing_graph(tmp_path, worker, helper)
    try:
        assert not reap_orphaned_workers(
            graph, ProcessAdapter(), ReapRequest(NodeWorkers(1), deadline=time.monotonic() + 3)
        )
        assert "interrupted observation=helper" in caplog.text
        assert worker.status() != psutil.STATUS_ZOMBIE
        recorded = graph.runs.get_worker("inv-1")
        assert recorded is not None and recorded.ended_at is None
    finally:
        graph.close()


def test_reap_observes_after_helper_commits_between_read_and_begin(
    tmp_path: Path, worker: psutil.Process
) -> None:
    current = psutil.Process()
    helper = HelperIdentity("inv-1", 0, current.pid, current.create_time())
    graph = _recorded_graph(tmp_path, worker, helper)

    def observe() -> None:
        graph.runs.begin_worker_observation(_helper_key(helper))
        graph.runs.commit_worker_observation(_helper_key(helper), ())

    try:
        assert _reap(graph, observe)
        assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE
        recorded = graph.runs.get_worker("inv-1")
        assert recorded is not None and recorded.ended_at is not None
        assert recorded.snapshot_seq == 2
    finally:
        graph.close()


def test_reap_accepts_helper_closure_between_read_and_begin(
    tmp_path: Path, worker: psutil.Process
) -> None:
    current = psutil.Process()
    helper = HelperIdentity("inv-1", 0, current.pid, current.create_time())
    graph = _recorded_graph(tmp_path, worker, helper)

    def close() -> None:
        graph.runs.begin_worker_observation(_helper_key(helper))
        graph.runs.commit_worker_observation(_helper_key(helper), ())
        graph.runs.end_worker("inv-1", 1, helper.generation)

    try:
        assert _reap(graph, close)
        recorded = graph.runs.get_worker("inv-1")
        assert recorded is not None and recorded.ended_at is not None
        assert recorded.snapshot_seq == 1
    finally:
        graph.close()


def test_reap_stays_unresolved_when_racing_helper_dies_mid_observation(
    tmp_path: Path, worker: psutil.Process, caplog: pytest.LogCaptureFixture
) -> None:
    helper = HelperIdentity("inv-1", 0, 2**31 - 1, 123.5)
    graph = _recorded_graph(tmp_path, worker, helper)
    try:
        assert not _reap(graph, lambda: graph.runs.begin_worker_observation(_helper_key(helper)))
        assert "worker recovery observation unresolved" in caplog.text
        assert worker.status() != psutil.STATUS_ZOMBIE
        recorded = graph.runs.get_worker("inv-1")
        assert recorded is not None and recorded.ended_at is None
        assert recorded.observation_owner == "helper"
    finally:
        graph.close()
