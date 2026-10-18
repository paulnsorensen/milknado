from __future__ import annotations

import time
from pathlib import Path

import pytest

import milknado.domains.dispatch.reap as reap
from milknado.domains.common import WorkerIdentity, WorkerOwner
from milknado.domains.dispatch import ReapRequest, WorkerCleanupResult, reap_orphaned_workers
from milknado.domains.graph import MikadoGraph, NodeWorkers

_OBSERVE_SECONDS = 0.6


class _SlowObserver:
    def terminate_group(self, pid: int, timeout: float) -> bool:
        raise AssertionError(f"unexpected supervisor cleanup: {pid} {timeout}")

    def supervisor_state(self, pid: int, start_token: float) -> str:
        assert (pid, start_token) == (999999, 123.5)
        return "gone"

    def observe_worker(self, worker: WorkerIdentity) -> tuple[tuple[int, float, int], ...]:
        assert worker.invocation_id == "inv-1"
        time.sleep(_OBSERVE_SECONDS)
        return ()

    def terminate_worker(
        self, worker: WorkerIdentity, retained: tuple[tuple[int, float, int], ...], deadline: float
    ) -> WorkerCleanupResult:
        assert (worker.invocation_id, retained) == ("inv-1", ())
        assert time.monotonic() < deadline
        return WorkerCleanupResult(True, ())


def _assert_slow_observation_releases_the_fence(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node = graph.add_node("recover")
        assert graph.claim_node(node.id, "run-1", now="2026-01-01T00:00:00+00:00", pid=999999)
        graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        graph.runs.record_worker(
            WorkerOwner("run-1", 999999, 123.5, "run-1", node.id),
            WorkerIdentity("inv-1", 999998, 999998, 123.5),
        )
        selection = NodeWorkers(node.id)
        process = _SlowObserver()
        first = reap_orphaned_workers(
            graph, process, ReapRequest(selection, deadline=time.monotonic() + 0.3)
        )
        assert not first
        records = graph.runs.live_workers(node_id=node.id)
        assert [record.observation_owner for record in records] == [None]
        assert reap_orphaned_workers(
            graph, process, ReapRequest(selection, deadline=time.monotonic() + 5)
        )
        assert graph.runs.live_workers(node_id=node.id) == ()
    finally:
        graph.close()


def test_observation_outlasting_the_deadline_does_not_wedge_later_recovery(
    tmp_path: Path,
) -> None:
    _assert_slow_observation_releases_the_fence(tmp_path)


def test_commit_window_starts_after_a_slow_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reap, "_COMMIT_SECONDS", _OBSERVE_SECONDS / 3)
    _assert_slow_observation_releases_the_fence(tmp_path)
