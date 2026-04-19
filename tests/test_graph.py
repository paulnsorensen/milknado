from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from milknado.domains.batching import (
    Batch,
    BatchPlan,
    SymbolRef,
    SymbolSpread,
)
from milknado.domains.common import NodeStatus
from milknado.domains.graph import MikadoGraph


class TestAddNode:
    def test_add_root_node(self, graph: MikadoGraph) -> None:
        node = graph.add_node("root goal")
        assert node.id == 1
        assert node.description == "root goal"
        assert node.status == NodeStatus.PENDING
        assert node.parent_id is None

    def test_add_child_node_creates_edge(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        assert child.parent_id == root.id
        children = graph.get_children(root.id)
        assert len(children) == 1
        assert children[0].id == child.id

    def test_add_multiple_children(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        c1 = graph.add_node("child-1", parent_id=root.id)
        c2 = graph.add_node("child-2", parent_id=root.id)
        children = graph.get_children(root.id)
        assert {c.id for c in children} == {c1.id, c2.id}


class TestAddEdge:
    def test_explicit_edge(self, graph: MikadoGraph) -> None:
        n1 = graph.add_node("a")
        n2 = graph.add_node("b")
        edge = graph.add_edge(n1.id, n2.id)
        assert edge.parent_id == n1.id
        assert edge.child_id == n2.id

    def test_cycle_detection_direct(self, graph: MikadoGraph) -> None:
        n1 = graph.add_node("a")
        n2 = graph.add_node("b")
        graph.add_edge(n1.id, n2.id)
        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge(n2.id, n1.id)

    def test_cycle_detection_transitive(self, graph: MikadoGraph) -> None:
        n1 = graph.add_node("a")
        n2 = graph.add_node("b")
        n3 = graph.add_node("c")
        graph.add_edge(n1.id, n2.id)
        graph.add_edge(n2.id, n3.id)
        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge(n3.id, n1.id)

    def test_self_loop_detected(self, graph: MikadoGraph) -> None:
        n = graph.add_node("self")
        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge(n.id, n.id)


class TestGetNode:
    def test_existing_node(self, graph: MikadoGraph) -> None:
        created = graph.add_node("find me")
        found = graph.get_node(created.id)
        assert found is not None
        assert found.description == "find me"

    def test_missing_node_returns_none(self, graph: MikadoGraph) -> None:
        assert graph.get_node(999) is None


class TestGetAllNodes:
    def test_empty_graph(self, graph: MikadoGraph) -> None:
        assert graph.get_all_nodes() == []

    def test_returns_all(self, graph: MikadoGraph) -> None:
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        assert len(graph.get_all_nodes()) == 3


class TestGetLeaves:
    def test_single_node_is_leaf(self, graph: MikadoGraph) -> None:
        node = graph.add_node("solo")
        leaves = graph.get_leaves()
        assert len(leaves) == 1
        assert leaves[0].id == node.id

    def test_children_are_leaves(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        c1 = graph.add_node("c1", parent_id=root.id)
        c2 = graph.add_node("c2", parent_id=root.id)
        leaves = graph.get_leaves()
        leaf_ids = {node.id for node in leaves}
        assert c1.id in leaf_ids
        assert c2.id in leaf_ids
        assert root.id not in leaf_ids


class TestGetRoot:
    def test_single_node_is_root(self, graph: MikadoGraph) -> None:
        node = graph.add_node("root")
        root = graph.get_root()
        assert root is not None
        assert root.id == node.id

    def test_root_with_children(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        graph.add_node("child", parent_id=root.id)
        found = graph.get_root()
        assert found is not None
        assert found.id == root.id

    def test_empty_graph_returns_none(self, graph: MikadoGraph) -> None:
        assert graph.get_root() is None


class TestGetReadyNodes:
    def test_pending_leaf_is_ready(self, graph: MikadoGraph) -> None:
        node = graph.add_node("ready")
        ready = graph.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == node.id

    def test_leaf_with_pending_status(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        ready = graph.get_ready_nodes()
        assert any(n.id == child.id for n in ready)

    def test_done_node_not_ready(self, graph: MikadoGraph) -> None:
        node = graph.add_node("ready")
        graph.mark_running(node.id)
        graph.mark_done(node.id)
        ready = graph.get_ready_nodes()
        assert all(n.id != node.id for n in ready)


class TestStatusTransitions:
    def test_pending_to_running(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_running(node.id)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.status == NodeStatus.RUNNING

    def test_running_to_done(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_running(node.id)
        graph.mark_done(node.id)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.status == NodeStatus.DONE
        assert updated.completed_at is not None

    def test_running_to_failed(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_running(node.id)
        graph.mark_failed(node.id)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.status == NodeStatus.FAILED

    def test_pending_to_blocked(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_blocked(node.id)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.status == NodeStatus.BLOCKED

    def test_blocked_to_pending_via_transition(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_blocked(node.id)
        graph._transition_status(node.id, NodeStatus.PENDING)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.status == NodeStatus.PENDING

    def test_failed_to_pending_via_transition(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_running(node.id)
        graph.mark_failed(node.id)
        graph._transition_status(node.id, NodeStatus.PENDING)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.status == NodeStatus.PENDING

    def test_invalid_transition_raises(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        with pytest.raises(ValueError, match="cannot transition"):
            graph.mark_done(node.id)

    def test_done_is_terminal(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_running(node.id)
        graph.mark_done(node.id)
        with pytest.raises(ValueError, match="cannot transition"):
            graph.mark_running(node.id)

    def test_running_to_pending_rollback(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_running(node.id, worktree_path="/tmp/wt", branch_name="b")
        graph.mark_pending(node.id)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.status == NodeStatus.PENDING
        assert updated.worktree_path is None
        assert updated.branch_name is None
        assert updated.run_id is None

    def test_mark_running_stores_run_id(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_running(node.id, worktree_path="/tmp/wt", run_id="run-42")
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.run_id == "run-42"

    def test_set_run_id(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_running(node.id)
        graph.set_run_id(node.id, "run-99")
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.run_id == "run-99"

    def test_set_run_id_nonexistent_raises(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.set_run_id(999, "run-99")

    def test_failed_clears_run_id(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.mark_running(node.id, run_id="run-1")
        graph.mark_failed(node.id)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.run_id is None

    def test_transition_nonexistent_node_raises(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.mark_running(999)


class TestFileOwnership:
    def test_set_and_get(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.set_file_ownership(node.id, ["a.py", "b.py"])
        files = graph.get_file_ownership(node.id)
        assert sorted(files) == ["a.py", "b.py"]

    def test_overwrite_ownership(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        graph.set_file_ownership(node.id, ["old.py"])
        graph.set_file_ownership(node.id, ["new.py"])
        assert graph.get_file_ownership(node.id) == ["new.py"]

    def test_empty_ownership(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task")
        assert graph.get_file_ownership(node.id) == []


class TestParallelSafety:
    def test_no_conflicts(self, graph: MikadoGraph) -> None:
        n1 = graph.add_node("a")
        n2 = graph.add_node("b")
        graph.set_file_ownership(n1.id, ["x.py"])
        graph.set_file_ownership(n2.id, ["y.py"])
        assert graph.check_parallel_safety([n1.id, n2.id]) == []

    def test_detects_overlap(self, graph: MikadoGraph) -> None:
        n1 = graph.add_node("a")
        n2 = graph.add_node("b")
        graph.set_file_ownership(n1.id, ["shared.py", "a.py"])
        graph.set_file_ownership(n2.id, ["shared.py", "b.py"])
        conflicts = graph.check_parallel_safety([n1.id, n2.id])
        assert len(conflicts) == 1
        assert conflicts[0][2] == ["shared.py"]

    def test_multiple_overlaps(self, graph: MikadoGraph) -> None:
        n1 = graph.add_node("a")
        n2 = graph.add_node("b")
        n3 = graph.add_node("c")
        graph.set_file_ownership(n1.id, ["shared.py"])
        graph.set_file_ownership(n2.id, ["shared.py"])
        graph.set_file_ownership(n3.id, ["shared.py"])
        conflicts = graph.check_parallel_safety([n1.id, n2.id, n3.id])
        assert len(conflicts) == 3


class TestClose:
    def test_close_prevents_further_ops(self, tmp_path: Path) -> None:
        g = MikadoGraph(tmp_path / "test.db")
        g.add_node("before close")
        g.close()
        with pytest.raises(Exception):
            g.add_node("after close")


class TestBatchMetadata:
    def test_default_fields_are_unset(self, graph: MikadoGraph) -> None:
        node = graph.add_node("plain")
        assert node.oversized is False
        assert node.batch_index is None
        fetched = graph.get_node(node.id)
        assert fetched is not None
        assert fetched.oversized is False
        assert fetched.batch_index is None

    def test_add_node_persists_oversized_and_batch_index(
        self, graph: MikadoGraph,
    ) -> None:
        node = graph.add_node("big batch", oversized=True, batch_index=2)
        assert node.oversized is True
        assert node.batch_index == 2
        fetched = graph.get_node(node.id)
        assert fetched is not None
        assert fetched.oversized is True
        assert fetched.batch_index == 2

    def test_set_batch_metadata_updates_existing_node(
        self, graph: MikadoGraph,
    ) -> None:
        node = graph.add_node("plain")
        graph.set_batch_metadata(node.id, oversized=True, batch_index=5)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.oversized is True
        assert updated.batch_index == 5

    def test_set_batch_metadata_clears_flags(self, graph: MikadoGraph) -> None:
        node = graph.add_node("big", oversized=True, batch_index=7)
        graph.set_batch_metadata(node.id, oversized=False, batch_index=None)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.oversized is False
        assert updated.batch_index is None

    def test_set_batch_metadata_unknown_node_raises(
        self, graph: MikadoGraph,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.set_batch_metadata(999, oversized=True, batch_index=0)


class TestBatchPlans:
    def test_empty_plan_roundtrip(self, graph: MikadoGraph) -> None:
        plan = BatchPlan(batches=(), spread_report=(), solver_status="OPTIMAL")
        plan_id = graph.record_batch_plan(plan)
        assert plan_id > 0
        latest = graph.get_latest_batch_plan()
        assert latest is not None
        assert latest["id"] == plan_id
        assert latest["solver_status"] == "OPTIMAL"
        assert latest["batch_count"] == 0
        assert latest["oversized_count"] == 0
        assert latest["max_spread"] == 0
        assert latest["spread_report"] == []

    def test_chained_plan_records_counts(self, graph: MikadoGraph) -> None:
        batches = (
            Batch(index=0, change_ids=("a",), depends_on=(), oversized=False),
            Batch(index=1, change_ids=("b",), depends_on=(0,), oversized=False),
        )
        spread = (
            SymbolSpread(symbol=SymbolRef(name="Foo", file="src/foo.py"), spread=1),
        )
        plan = BatchPlan(
            batches=batches, spread_report=spread, solver_status="OPTIMAL",
        )
        graph.record_batch_plan(plan)
        latest = graph.get_latest_batch_plan()
        assert latest is not None
        assert latest["batch_count"] == 2
        assert latest["oversized_count"] == 0
        assert latest["max_spread"] == 1
        assert latest["spread_report"] == [
            {"symbol_name": "Foo", "symbol_file": "src/foo.py", "spread": 1},
        ]

    def test_oversized_plan_flags_count(self, graph: MikadoGraph) -> None:
        batches = (
            Batch(
                index=0,
                change_ids=("big",),
                depends_on=(),
                oversized=True,
            ),
        )
        plan = BatchPlan(
            batches=batches, spread_report=(), solver_status="FEASIBLE",
        )
        graph.record_batch_plan(plan)
        latest = graph.get_latest_batch_plan()
        assert latest is not None
        assert latest["solver_status"] == "FEASIBLE"
        assert latest["batch_count"] == 1
        assert latest["oversized_count"] == 1
        assert latest["max_spread"] == 0

    def test_get_latest_returns_none_when_empty(
        self, graph: MikadoGraph,
    ) -> None:
        assert graph.get_latest_batch_plan() is None

    def test_get_latest_returns_most_recent(self, graph: MikadoGraph) -> None:
        graph.record_batch_plan(
            BatchPlan(batches=(), spread_report=(), solver_status="OPTIMAL"),
        )
        second = graph.record_batch_plan(
            BatchPlan(batches=(), spread_report=(), solver_status="FEASIBLE"),
        )
        latest = graph.get_latest_batch_plan()
        assert latest is not None
        assert latest["id"] == second
        assert latest["solver_status"] == "FEASIBLE"


class TestSchemaMigration:
    def test_adds_run_id_column_for_existing_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parent_id INTEGER,
                worktree_path TEXT,
                branch_name TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE edges (
                parent_id INTEGER NOT NULL,
                child_id INTEGER NOT NULL,
                PRIMARY KEY (parent_id, child_id)
            );
            CREATE TABLE file_ownership (
                node_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                PRIMARY KEY (node_id, file_path)
            );
        """)
        conn.execute(
            "INSERT INTO nodes (description, status, created_at) VALUES (?, ?, ?)",
            ("legacy node", "pending", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        graph = MikadoGraph(db_path)
        node = graph.get_node(1)
        assert node is not None
        assert node.run_id is None
        graph.close()
