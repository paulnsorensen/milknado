from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.graph import (
    MikadoGraph,
    RunWorkers,
    UnassociatedWorkers,
    WorkerEvidenceStore,
    open_standalone_worker_evidence,
)


def test_worker_record_persists_and_blocks_reclaim(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    node = graph.add_node("worker")
    assert graph.claim_node(node.id, "run-1", now="2026-01-01T00:00:00+00:00", pid=999999)
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    graph.runs.record_worker(
        WorkerOwner("run-1", 999999, 123.5, "run-1", node.id),
        WorkerIdentity("inv-1", 2345, 2345, 123.5),
    )
    graph.close()

    graph = MikadoGraph(db_path)
    try:
        workers = graph.runs.live_workers(run_id="run-1")
        assert len(workers) == 1
        assert workers[0].node_id == node.id
        assert workers[0].pid == 2345
        assert workers[0].start_token == 123.5
        assert workers[0].ended_at is None
        assert graph.try_reclaim(node.id, now="2026-01-01T00:01:00+00:00") is False
        current = graph.get_node(node.id)
        assert current is not None and current.run_id == "run-1"
    finally:
        graph.close()


def test_reclaim_guard_covers_worker_run_under_parent_owner(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node = graph.add_node("worker")
        assert graph.claim_node(node.id, "parent", now="2026-01-01T00:00:00+00:00", pid=999999)
        graph.runs.start("child", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        graph.runs.record_worker(
            WorkerOwner("child", 999999, 123.5, "child", node.id),
            WorkerIdentity("inv-child", 2345, 2345, 123.5),
        )
        assert graph.try_reclaim(node.id, now="2026-01-01T00:01:00+00:00") is False
        current = graph.get_node(node.id)
        assert current is not None and current.run_id == "parent"
    finally:
        graph.close()


def test_worker_observation_and_helper_ready_are_fenced(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node = graph.add_node("worker")
        graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        graph.runs.record_worker(
            WorkerOwner("run-1", 999999, 123.5, "run-1", node.id),
            WorkerIdentity("inv-1", 2345, 2345, 123.5),
        )
        helper = HelperIdentity("inv-1", 0, 3456, 234.5)
        observation = ObservationKey("inv-1", "supervisor", 1, -1, 2345, 123.5)
        graph.runs.record_helper(helper)
        graph.runs.begin_worker_observation(observation)
        assert graph.runs.ready_helper(helper, 0) is False
        graph.runs.commit_worker_observation(observation, ((4567, 345.5, 4567),))
        assert graph.runs.ready_helper(helper, 0) is False
        assert graph.runs.ready_helper(helper, 1) is True
        assert graph.runs.ready_helper(HelperIdentity("inv-1", 0, 3456, 999.0), 1) is False
        workers = graph.runs.live_workers(run_id="run-1")
        assert workers[0].descendants == ((4567, 345.5, 4567),)
    finally:
        graph.close()


def test_interrupted_observation_prevents_worker_closure(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node = graph.add_node("worker")
        graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        graph.runs.record_worker(
            WorkerOwner("run-1", 999999, 123.5, "run-1", node.id),
            WorkerIdentity("inv-1", 2345, 2345, 123.5),
        )
        graph.runs.begin_worker_observation(
            ObservationKey("inv-1", "supervisor", 1, -1, 2345, 123.5)
        )
        with pytest.raises(RuntimeError, match="observation"):
            graph.runs.end_worker("inv-1", 0)
        assert graph.runs.live_workers(run_id="run-1")[0].observation_owner == "supervisor"
    finally:
        graph.close()


def test_replaced_helper_cannot_begin_or_commit_observation(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node = graph.add_node("worker")
        graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        graph.runs.record_worker(
            WorkerOwner("run-1", 999999, 123.5, "run-1", node.id),
            WorkerIdentity("inv-1", 2345, 2345, 123.5),
        )
        old = HelperIdentity("inv-1", 0, 3456, 234.5)
        graph.runs.record_helper(old)
        graph.runs.record_helper(HelperIdentity("inv-1", 1, 4567, 345.5))
        stale = ObservationKey("inv-1", "helper", 1, 0, old.pid, old.start_token)
        with pytest.raises(RuntimeError, match="observation"):
            graph.runs.begin_worker_observation(stale)
        current = ObservationKey("inv-1", "helper", 1, 1, 4567, 345.5)
        graph.runs.begin_worker_observation(current)
        with pytest.raises(RuntimeError, match="observation"):
            graph.runs.commit_worker_observation(stale, ((5678, 456.5, 5678),))
        graph.runs.commit_worker_observation(current, ((5678, 456.5, 5678),))
        assert graph.runs.live_workers(run_id="run-1")[0].descendants == ((5678, 456.5, 5678),)
    finally:
        graph.close()


def test_stale_verified_snapshot_cannot_close_newer_evidence(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node = graph.add_node("worker")
        graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        graph.runs.record_worker(
            WorkerOwner("run-1", 999999, 123.5, "run-1", node.id),
            WorkerIdentity("inv-1", 2345, 2345, 123.5),
        )
        first = ObservationKey("inv-1", "supervisor", 1, -1, 2345, 123.5)
        graph.runs.begin_worker_observation(first)
        graph.runs.commit_worker_observation(first, ((4567, 345.5, 4567),))
        with WorkerEvidenceStore(graph.db_path) as other:
            newer = ObservationKey("inv-1", "supervisor", 2, -1, 2345, 123.5)
            other.begin(newer)
            other.commit(newer, ((5678, 456.5, 5678),))
        with pytest.raises(RuntimeError, match="snapshot"):
            graph.runs.end_worker("inv-1", 1)
        record = graph.runs.get_worker("inv-1")
        assert record is not None
        assert record.ended_at is None
        assert record.descendants == ((4567, 345.5, 4567), (5678, 456.5, 5678))
        graph.runs.end_worker("inv-1", 2)
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        graph.close()


def test_worker_evidence_connection_respects_one_deadline_under_writer_lock(
    tmp_path: Path,
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    graph.runs.record_worker(
        WorkerOwner("run-1", 999999, 123.5, "run-1", node.id),
        WorkerIdentity("inv-1", 2345, 2345, 123.5),
    )
    lock = sqlite3.connect(graph.db_path)
    try:
        lock.execute("BEGIN IMMEDIATE")
        lock.execute(
            "UPDATE run_workers SET snapshot_seq = snapshot_seq WHERE invocation_id = ?",
            ("inv-1",),
        )
        start = time.monotonic()
        with WorkerEvidenceStore(graph.db_path, deadline=start + 0.3) as store:
            assert len(store.live_workers(RunWorkers("run-1"))) == 1
            with pytest.raises((sqlite3.OperationalError, TimeoutError)):
                store.begin(ObservationKey("inv-1", "supervisor", 1, -1, 2345, 123.5))
        assert time.monotonic() - start < 0.8
    finally:
        lock.rollback()
        lock.close()
        graph.close()


def test_unassociated_worker_uses_same_durable_store(tmp_path: Path) -> None:
    with open_standalone_worker_evidence(tmp_path / "workers.db") as store:
        store.record_worker(
            WorkerOwner("runtime-1", 999999, 123.5),
            WorkerIdentity("inv-1", 2345, 2345, 123.5),
        )
        records = store.live_workers(UnassociatedWorkers())
        assert len(records) == 1
        assert records[0].runtime_run_id == "runtime-1"
        assert records[0].graph_run_id is None
        assert records[0].node_id is None
        assert records[0].supervisor_pid == 999999