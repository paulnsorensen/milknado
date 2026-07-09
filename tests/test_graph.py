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
from milknado.domains.common import InvalidTransition, NodeKind, NodeStatus
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

    def test_add_node_rejects_nonexistent_parent_without_persisting(
        self, graph: MikadoGraph
    ) -> None:
        # The node row commits before the edge, so a missing parent must be
        # rejected up front rather than leaving a committed stray node.
        with pytest.raises(ValueError, match="parent_id 999 not found"):
            graph.add_node("orphan", parent_id=999)
        assert graph.get_all_nodes() == []


class TestDeleteNode:
    def test_delete_leaf_removes_it(self, graph: MikadoGraph) -> None:
        node = graph.add_node("leaf")
        assert graph.delete_node(node.id) == 1
        assert graph.get_node(node.id) is None

    def test_delete_leaf_removes_file_ownership(self, graph: MikadoGraph) -> None:
        node = graph.add_node("leaf")
        graph.set_file_ownership(node.id, ["src/a.py"])
        graph.delete_node(node.id)
        assert graph.get_file_ownership(node.id) == []

    def test_delete_node_with_children_without_cascade_raises(self, graph: MikadoGraph) -> None:
        parent = graph.add_node("parent")
        graph.add_node("child", parent_id=parent.id)
        with pytest.raises(ValueError, match="cascade"):
            graph.delete_node(parent.id)
        assert graph.get_node(parent.id) is not None

    def test_delete_cascade_removes_whole_subtree(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        grandchild = graph.add_node("grandchild", parent_id=child.id)
        assert graph.delete_node(root.id, cascade=True) == 3
        assert graph.get_node(root.id) is None
        assert graph.get_node(child.id) is None
        assert graph.get_node(grandchild.id) is None

    def test_delete_cascade_diamond_deletes_shared_descendant_once(
        self, graph: MikadoGraph
    ) -> None:
        # Diamond: root -> {left, right}; both left and right -> shared.
        # The shared descendant is reachable via two paths; cascade must delete
        # it exactly once rather than re-visiting it and raising mid-cascade.
        root = graph.add_node("root")
        left = graph.add_node("left", parent_id=root.id)
        right = graph.add_node("right", parent_id=root.id)
        shared = graph.add_node("shared", parent_id=left.id)
        graph.add_edge(right.id, shared.id)

        assert graph.delete_node(root.id, cascade=True) == 4
        for node in (root, left, right, shared):
            assert graph.get_node(node.id) is None

    def test_delete_unknown_node_raises(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.delete_node(999)

    def test_delete_nulls_dangling_parent_id_of_non_edge_child(self, graph: MikadoGraph) -> None:
        # set_parent_id points parent_id at a node WITHOUT creating an edge
        # (batching bridge). Deleting that parent must null the orphan's
        # parent_id so no row references a deleted node.
        parent = graph.add_node("parent")
        orphan = graph.add_node("orphan")
        graph.set_parent_id(orphan.id, parent.id)
        graph.delete_node(parent.id)
        refreshed = graph.get_node(orphan.id)
        assert refreshed is not None
        assert refreshed.parent_id is None


class TestUpdateNode:
    def test_update_description(self, graph: MikadoGraph) -> None:
        node = graph.add_node("old")
        graph.update_node(node.id, description="new")
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.description == "new"

    def test_update_kind(self, graph: MikadoGraph) -> None:
        from milknado.domains.common import NodeKind

        node = graph.add_node("n")
        graph.update_node(node.id, kind=NodeKind.GOAL)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.kind == NodeKind.GOAL

    def test_update_both_fields_at_once(self, graph: MikadoGraph) -> None:
        from milknado.domains.common import NodeKind

        node = graph.add_node("old")
        graph.update_node(node.id, description="new", kind=NodeKind.GOAL)
        updated = graph.get_node(node.id)
        assert updated is not None
        assert updated.description == "new"
        assert updated.kind == NodeKind.GOAL

    def test_update_unknown_node_raises(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.update_node(999, description="x")

    def test_update_with_no_fields_raises(self, graph: MikadoGraph) -> None:
        node = graph.add_node("n")
        with pytest.raises(ValueError, match="nothing to update"):
            graph.update_node(node.id)


class TestMoveNode:
    def test_move_rewrites_edge_and_parent_id(self, graph: MikadoGraph) -> None:
        old_parent = graph.add_node("old")
        new_parent = graph.add_node("new")
        child = graph.add_node("child", parent_id=old_parent.id)

        graph.move_node(child.id, new_parent.id)

        moved = graph.get_node(child.id)
        assert moved is not None
        assert moved.parent_id == new_parent.id
        assert [c.id for c in graph.get_children(old_parent.id)] == []
        assert [c.id for c in graph.get_children(new_parent.id)] == [child.id]

    def test_move_to_root_drops_edge(self, graph: MikadoGraph) -> None:
        parent = graph.add_node("p")
        child = graph.add_node("c", parent_id=parent.id)

        graph.move_node(child.id, None)

        moved = graph.get_node(child.id)
        assert moved is not None
        assert moved.parent_id is None
        assert graph.get_children(parent.id) == []

    def test_move_under_grandchild_raises_cycle(self, graph: MikadoGraph) -> None:
        """A multi-level descendant must also be rejected, not just direct children."""
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        grandchild = graph.add_node("grandchild", parent_id=child.id)

        with pytest.raises(ValueError, match="cycle"):
            graph.move_node(root.id, grandchild.id)

    def test_move_unknown_node_raises(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.move_node(999, None)

    def test_move_to_unknown_parent_raises(self, graph: MikadoGraph) -> None:
        node = graph.add_node("n")
        with pytest.raises(ValueError, match="not found"):
            graph.move_node(node.id, 999)

    def test_move_under_node_with_diamond_ancestry_allows_non_cycle(
        self, graph: MikadoGraph
    ) -> None:
        """Cycle walk dedups the new parent's shared ancestors (diamond) yet still allows
        a legal move."""
        top = graph.add_node("top")
        left = graph.add_node("left", parent_id=top.id)
        right = graph.add_node("right", parent_id=top.id)
        mid = graph.add_node("mid", parent_id=left.id)
        graph.add_edge(right.id, mid.id)
        loose = graph.add_node("loose")

        # Re-parenting loose under mid walks mid's ancestors: top is reachable via
        # both left and right, so the visited-set guard prevents reprocessing it.
        graph.move_node(loose.id, mid.id)
        moved = graph.get_node(loose.id)
        assert moved is not None
        assert moved.parent_id == mid.id


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

    def test_get_root_deterministic_in_forest(self, graph: MikadoGraph) -> None:
        """get_root returns the lowest-id node when multiple roots exist."""
        a = graph.add_node("first")
        b = graph.add_node("second")
        root = graph.get_root()
        assert root is not None
        assert root.id == min(a.id, b.id)


class TestGetReadyNodes:
    def test_pending_leaf_is_ready(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        leaf = graph.add_node("leaf", parent_id=root.id)
        ready = graph.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == leaf.id

    def test_leaf_with_pending_status(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        ready = graph.get_ready_nodes()
        assert any(n.id == child.id for n in ready)

    def test_done_node_not_ready(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        leaf = graph.add_node("leaf", parent_id=root.id)
        graph.mark_running(leaf.id)
        graph.mark_done(leaf.id)
        ready = graph.get_ready_nodes()
        assert all(n.id != leaf.id for n in ready)

    def test_root_excluded_when_all_children_done(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        graph.mark_running(child.id)
        graph.mark_done(child.id)
        ready = graph.get_ready_nodes()
        assert all(n.id != root.id for n in ready)
        assert len(ready) == 0

    def test_leaf_ready_but_root_excluded(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        leaf = graph.add_node("leaf", parent_id=root.id)
        ready = graph.get_ready_nodes()
        ready_ids = {n.id for n in ready}
        assert leaf.id in ready_ids
        assert root.id not in ready_ids

    def test_solo_root_not_ready(self, graph: MikadoGraph) -> None:
        # A single root node with no children is the root — it must not be ready.
        root = graph.add_node("solo root")
        ready = graph.get_ready_nodes()
        assert all(n.id != root.id for n in ready)
        assert len(ready) == 0

    def test_all_roots_excluded_in_forest(self, graph: MikadoGraph) -> None:
        # Every root must be excluded, not just one arbitrary get_root() pick —
        # otherwise a second childless root leaks in as "ready".
        r1 = graph.add_node("root1")
        r2 = graph.add_node("root2")
        leaf = graph.add_node("leaf", parent_id=r1.id)
        ready_ids = {n.id for n in graph.get_ready_nodes()}
        assert r1.id not in ready_ids
        assert r2.id not in ready_ids
        assert leaf.id in ready_ids

    def test_get_ready_nodes_single_edges_scan(self, tmp_path: Path) -> None:
        """get_ready_nodes issues O(1) queries regardless of tree size (was N+1)."""
        g = MikadoGraph(tmp_path / "qcount.db")
        root = g.add_node("root")
        for i in range(20):
            g.add_node(f"child {i}", parent_id=root.id)

        queries: list[str] = []
        g._conn.set_trace_callback(queries.append)
        result = g.get_ready_nodes()
        g._conn.set_trace_callback(None)
        g.close()

        edges_queries = [q for q in queries if "edges" in q]
        assert len(edges_queries) <= 3, f"expected <=3 edges scans, got {len(edges_queries)}"
        assert len(result) == 20


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
        with pytest.raises(InvalidTransition, match="cannot transition") as exc_info:
            graph.mark_done(node.id)
        exc = exc_info.value
        assert exc.node_id == node.id
        assert exc.current == NodeStatus.PENDING
        assert exc.target == NodeStatus.DONE
        assert set(exc.valid_targets) == {
            NodeStatus.RUNNING,
            NodeStatus.BLOCKED,
            NodeStatus.FAILED,
        }

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
        self,
        graph: MikadoGraph,
    ) -> None:
        node = graph.add_node("big batch", oversized=True, batch_index=2)
        assert node.oversized is True
        assert node.batch_index == 2
        fetched = graph.get_node(node.id)
        assert fetched is not None
        assert fetched.oversized is True
        assert fetched.batch_index == 2

    def test_set_batch_metadata_updates_existing_node(
        self,
        graph: MikadoGraph,
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
        self,
        graph: MikadoGraph,
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
        spread = (SymbolSpread(symbol=SymbolRef(name="Foo", file="src/foo.py"), spread=1),)
        plan = BatchPlan(
            batches=batches,
            spread_report=spread,
            solver_status="OPTIMAL",
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
            batches=batches,
            spread_report=(),
            solver_status="FEASIBLE",
        )
        graph.record_batch_plan(plan)
        latest = graph.get_latest_batch_plan()
        assert latest is not None
        assert latest["solver_status"] == "FEASIBLE"
        assert latest["batch_count"] == 1
        assert latest["oversized_count"] == 1
        assert latest["max_spread"] == 0

    def test_get_latest_returns_none_when_empty(
        self,
        graph: MikadoGraph,
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


class TestDeleteSubtreePostOrder:
    def test_collect_post_order_children_before_parent(self) -> None:
        from milknado.domains.graph._mutations import _collect_subtree_post_order

        children_map = {1: [2, 4], 2: [3]}
        result = _collect_subtree_post_order(children_map, 1)
        assert result.index(3) < result.index(2), "grandchild must precede child"
        assert result.index(2) < result.index(1), "child must precede root"
        assert result.index(4) < result.index(1), "sibling must precede root"
        assert set(result) == {1, 2, 3, 4}

    def test_collect_post_order_diamond_visits_each_once(self) -> None:
        from milknado.domains.graph._mutations import _collect_subtree_post_order

        children_map = {1: [2, 3], 2: [4], 3: [4]}
        result = _collect_subtree_post_order(children_map, 1)
        assert result.count(4) == 1, "shared descendant visited exactly once"
        assert result.index(4) < result.index(2)
        assert result.index(4) < result.index(3)
        assert result.index(2) < result.index(1)
        assert result.index(3) < result.index(1)

    def test_delete_subtree_removes_all_nodes_atomically(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        graph.add_node("grandchild", parent_id=child.id)
        count = graph.delete_node(root.id, cascade=True)
        assert count == 3
        assert graph.get_all_nodes() == []

    def test_delete_subtree_leaves_no_stale_edges(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        graph.add_node("grandchild", parent_id=child.id)
        graph.delete_node(root.id, cascade=True)
        rows = graph._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert rows == 0

    def test_delete_subtree_rolls_back_on_error(self, graph: MikadoGraph) -> None:
        """If a cascade delete fails mid-way, the whole subtree survives."""
        from unittest.mock import patch

        from milknado.domains.graph._mutations import _delete_one as real_delete_one

        root = graph.add_node("root")
        graph.add_node("child1", parent_id=root.id)
        graph.add_node("child2", parent_id=root.id)

        call_count = {"n": 0}

        def fail_on_second(conn, nid: int) -> None:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("injected failure")
            real_delete_one(conn, nid)

        with patch("milknado.domains.graph._mutations._delete_one", side_effect=fail_on_second):
            with pytest.raises(RuntimeError, match="injected failure"):
                graph.delete_node(root.id, cascade=True)

        assert len(graph.get_all_nodes()) == 3


class TestFlavorRegistry:
    def test_add_node_accepts_flavor_in_default_registry(self, graph: MikadoGraph) -> None:
        node = graph.add_node("t", flavor="spike")
        assert node.flavor == "spike"

    def test_add_node_rejects_flavor_outside_default_registry(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="unknown flavor"):
            graph.add_node("t", flavor="brainstorm")

    def test_add_node_accepts_custom_flavor_via_registry_param(self, graph: MikadoGraph) -> None:
        node = graph.add_node(
            "t", flavor="brainstorm", flavor_registry=frozenset({"implement", "brainstorm"})
        )
        assert node.flavor == "brainstorm"

    def test_add_node_rejects_flavor_not_in_custom_registry(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="unknown flavor"):
            graph.add_node("t", flavor="spike", flavor_registry=frozenset({"implement"}))

    def test_unknown_flavor_error_names_flavor_and_valid_set(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match=r"'brainstorm'.*'implement'"):
            graph.add_node("t", flavor="brainstorm", flavor_registry=frozenset({"implement"}))


class TestArtifactPath:
    def test_add_node_persists_artifact_path(self, graph: MikadoGraph) -> None:
        node = graph.add_node("t", artifact_path="docs/notes/t.md")
        assert node.artifact_path == "docs/notes/t.md"
        reloaded = graph.get_node(node.id)
        assert reloaded is not None
        assert reloaded.artifact_path == "docs/notes/t.md"

    def test_add_node_defaults_artifact_path_to_none(self, graph: MikadoGraph) -> None:
        node = graph.add_node("t")
        assert node.artifact_path is None

    def test_update_node_sets_artifact_path(self, graph: MikadoGraph) -> None:
        node = graph.add_node("t")
        graph.update_node(node.id, artifact_path="docs/notes/t.md")
        reloaded = graph.get_node(node.id)
        assert reloaded is not None
        assert reloaded.artifact_path == "docs/notes/t.md"


class TestPrereqEdges:
    def test_add_node_creates_prereq_edges(self, graph: MikadoGraph) -> None:
        goal = graph.add_node("goal", kind=NodeKind.GOAL)
        p1 = graph.add_node("prereq-1", parent_id=goal.id)
        p2 = graph.add_node("prereq-2", parent_id=goal.id)
        node = graph.add_node("t", parent_id=goal.id, prereqs=[p1.id, p2.id])
        parents = {
            row[0]
            for row in graph._conn.execute(
                "SELECT parent_id FROM edges WHERE child_id = ?", (node.id,)
            ).fetchall()
        }
        assert parents == {goal.id, p1.id, p2.id}

    def test_add_node_prereq_duplicating_containment_edge_raises(self, graph: MikadoGraph) -> None:
        """A prereq id equal to parent_id re-inserts the same (parent, child) edge,
        which the edges table's composite primary key refuses."""
        root = graph.add_node("root")
        before = len(graph.get_all_nodes())
        with pytest.raises(ValueError):
            graph.add_node("t", parent_id=root.id, prereqs=[root.id])
        assert len(graph.get_all_nodes()) == before

    def test_add_node_prereq_missing_node_raises(self, graph: MikadoGraph) -> None:
        before = len(graph.get_all_nodes())
        with pytest.raises(ValueError):
            graph.add_node("t", prereqs=[999])
        assert len(graph.get_all_nodes()) == before
