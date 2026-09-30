from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pytest

from milknado.domains.common import ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.dispatch import ReapRequest, WorkerCleanupResult, reap_orphaned_workers
from milknado.domains.graph import (
    MikadoGraph,
    NodeWorkers,
    UnassociatedWorkers,
    WorkerEvidenceStore,
)


class _ProcessBoundary:
    def __init__(self, graph: MikadoGraph, mode: Literal["observe", "cleanup", "closure"]) -> None:
        self.graph: MikadoGraph = graph
        self.mode: Literal["observe", "cleanup", "closure"] = mode
        self.signals: int = 0

    def terminate_group(self, pid: int, timeout: float) -> bool:
        raise AssertionError(f"unexpected supervisor cleanup: {pid} {timeout}")

    def supervisor_state(self, pid: int, start_token: float) -> str:
        raise AssertionError(f"unexpected supervisor inspection: {pid} {start_token}")

    def observe_worker(self, worker: WorkerIdentity) -> tuple[tuple[int, float, int], ...]:
        if self.mode == "observe":
            raise RuntimeError(f"enumeration unavailable for {worker.invocation_id}")
        assert worker.invocation_id == "inv-1"
        return ()

    def terminate_worker(
        self, worker: WorkerIdentity, retained: tuple[tuple[int, float, int], ...], deadline: float
    ) -> WorkerCleanupResult:
        assert retained == ()
        assert time.monotonic() < deadline
        self.signals += 1
        if self.mode == "cleanup":
            raise RuntimeError("signal unavailable")
        self.graph.runs.begin_worker_observation(
            ObservationKey(
                worker.invocation_id, "supervisor", 2, -1, worker.pid, worker.start_token
            )
        )
        return WorkerCleanupResult(True, ())


def _graph(tmp_path: Path) -> MikadoGraph:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("recover")
    assert graph.claim_node(node.id, "run-1", now="2026-01-01T00:00:00+00:00", pid=999999)
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    graph.runs.record_worker(
        WorkerOwner("run-1", 999999, 123.5, "run-1", node.id),
        WorkerIdentity("inv-1", 999998, 999998, 123.5),
    )
    return graph


@pytest.mark.parametrize("mode", ["observe", "cleanup", "closure"])
def test_failed_recovery_keeps_worker_record_and_node_owned(
    tmp_path: Path, mode: Literal["observe", "cleanup", "closure"]
) -> None:
    graph = _graph(tmp_path)
    process = _ProcessBoundary(graph, mode)
    try:
        assert not reap_orphaned_workers(
            graph,
            process,
            ReapRequest(NodeWorkers(1), deadline=time.monotonic() + 2),
        )
        records = graph.runs.live_workers(node_id=1)
        assert len(records) == 1
        assert records[0].ended_at is None
        assert records[0].observation_owner == ("supervisor" if mode != "cleanup" else None)
        assert process.signals == (0 if mode == "observe" else 1)
        assert not graph.try_reclaim(1, now="2026-01-01T00:01:00+00:00")
    finally:
        graph.close()


class _OwnerBoundary:
    def __init__(self, state: Literal["live", "unknown"]) -> None:
        self.state: Literal["live", "unknown"] = state

    def terminate_group(self, pid: int, timeout: float) -> bool:
        raise AssertionError(f"unexpected supervisor cleanup: {pid} {timeout}")

    def supervisor_state(self, pid: int, start_token: float) -> str:
        assert (pid, start_token) == (999999, 123.5)
        return self.state

    def observe_worker(self, worker: WorkerIdentity) -> tuple[tuple[int, float, int], ...]:
        raise AssertionError(f"unexpected observation: {worker.invocation_id}")

    def terminate_worker(
        self, worker: WorkerIdentity, retained: tuple[tuple[int, float, int], ...], deadline: float
    ) -> WorkerCleanupResult:
        raise AssertionError(f"unexpected signal: {worker.invocation_id} {retained} {deadline}")


@pytest.mark.parametrize("state", ["live", "unknown"])
def test_unassociated_worker_stays_open_without_dead_owner_proof(
    tmp_path: Path, state: Literal["live", "unknown"]
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with WorkerEvidenceStore(graph.db_path) as store:
            store.record_worker(
                WorkerOwner("runtime-1", 999999, 123.5),
                WorkerIdentity("inv-1", 999998, 999998, 123.5),
            )
        recovered = reap_orphaned_workers(
            graph, _OwnerBoundary(state), ReapRequest(UnassociatedWorkers())
        )
        assert recovered is (state == "live")
        with WorkerEvidenceStore(graph.db_path) as store:
            records = store.live_workers(UnassociatedWorkers())
        assert len(records) == 1
        assert records[0].ended_at is None
        assert records[0].observation_owner is None
    finally:
        graph.close()
