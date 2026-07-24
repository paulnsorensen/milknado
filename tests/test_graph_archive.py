"""Archive/unarchive lifecycle rules for DONE subtrees (spec: archive-rebalance).

Archive is a reversible soft-hide via the `archived_at` timestamp (migration
v3), NOT a status. These tests pin the lifecycle acceptance rules: eligibility
is fail-loud, unarchive clears the node and its archived descendants but
refuses under an archived ancestor (over the edge relation, diamonds
included), archived nodes refuse status writes and new children, bulk subtree
status writes skip them, and delete_node stays a hard delete.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from milknado.domains.common import ArchiveIneligible, MikadoNode, NodeKind, NodeSpec, NodeStatus
from milknado.domains.common.errors import InvalidTransition
from milknado.domains.graph import MikadoGraph, _mutations, _status
from milknado.domains.graph._goal_claims import get_goal_claim
from milknado.domains.graph._persistence import SCHEMA_VERSION
from milknado.domains.graph._pipeline import StatusPipeline

STAMP = "2026-07-24T00:00:00+00:00"


def _complete(graph: MikadoGraph, node_id: int) -> None:
    graph.mark_running(node_id)
    graph.mark_done(node_id)


def _node(graph: MikadoGraph, node_id: int) -> MikadoNode:
    node = graph.get_node(node_id)
    assert node is not None
    return node


def _done_goal_with_two_tasks(graph: MikadoGraph) -> tuple[MikadoNode, MikadoNode, MikadoNode]:
    goal = graph.add_node("goal")
    left = graph.add_node("left", parent_id=goal.id)
    right = graph.add_node("right", parent_id=goal.id)
    for nid in (left.id, right.id, goal.id):
        _complete(graph, nid)
    return goal, left, right


class TestArchiveSubtree:
    def test_all_done_subtree_stamps_every_node_and_returns_count(
        self, graph: MikadoGraph
    ) -> None:
        goal, left, right = _done_goal_with_two_tasks(graph)
        assert _mutations.archive_subtree(graph._conn, goal.id, now=STAMP) == 3
        for nid in (goal.id, left.id, right.id):
            assert _node(graph, nid).archived_at == datetime.fromisoformat(STAMP)

    def test_defaults_to_utcnow_stamp(self, graph: MikadoGraph) -> None:
        node = graph.add_node("done")
        _complete(graph, node.id)
        assert _mutations.archive_subtree(graph._conn, node.id) == 1
        stamped = _node(graph, node.id).archived_at
        assert stamped is not None
        assert abs((datetime.now(UTC) - stamped).total_seconds()) < 60

    def test_mixed_status_subtree_raises_naming_ids_and_stamps_nothing(
        self, graph: MikadoGraph
    ) -> None:
        goal = graph.add_node("goal")
        done_child = graph.add_node("done", parent_id=goal.id)
        pending_child = graph.add_node("pending", parent_id=goal.id)
        running_child = graph.add_node("running", parent_id=goal.id)
        graph.mark_running(running_child.id)
        _complete(graph, done_child.id)
        _complete(graph, goal.id)
        with pytest.raises(ArchiveIneligible) as excinfo:
            _mutations.archive_subtree(graph._conn, goal.id)
        assert excinfo.value.node_ids == (pending_child.id, running_child.id)
        for nid in (goal.id, done_child.id, pending_child.id, running_child.id):
            assert _node(graph, nid).archived_at is None

    def test_archive_missing_node_raises(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="Node 999 not found"):
            _mutations.archive_subtree(graph._conn, 999)

    def test_archive_non_done_node_raises_naming_itself(self, graph: MikadoGraph) -> None:
        node = graph.add_node("pending")
        with pytest.raises(ArchiveIneligible) as excinfo:
            _mutations.archive_subtree(graph._conn, node.id)
        assert excinfo.value.node_ids == (node.id,)
        assert str(node.id) in str(excinfo.value)
        assert _node(graph, node.id).archived_at is None

    def test_rearchive_is_idempotent_and_restamps(self, graph: MikadoGraph) -> None:
        goal, left, right = _done_goal_with_two_tasks(graph)
        assert _mutations.archive_subtree(graph._conn, goal.id, now=STAMP) == 3
        later = "2026-07-24T01:00:00+00:00"
        assert _mutations.archive_subtree(graph._conn, goal.id, now=later) == 3
        for nid in (goal.id, left.id, right.id):
            assert _node(graph, nid).archived_at == datetime.fromisoformat(later)

    def test_archive_subtree_containing_already_archived_descendant_restamps_all(
        self, graph: MikadoGraph
    ) -> None:
        goal, left, right = _done_goal_with_two_tasks(graph)
        _mutations.archive_subtree(graph._conn, left.id, now=STAMP)
        later = "2026-07-24T01:00:00+00:00"
        assert _mutations.archive_subtree(graph._conn, goal.id, now=later) == 3
        for nid in (goal.id, left.id, right.id):
            assert _node(graph, nid).archived_at == datetime.fromisoformat(later)


class TestUnarchiveSubtree:
    def test_unarchive_from_archived_root_clears_whole_subtree(self, graph: MikadoGraph) -> None:
        goal, left, right = _done_goal_with_two_tasks(graph)
        _mutations.archive_subtree(graph._conn, goal.id)
        assert _mutations.unarchive_subtree(graph._conn, goal.id) == 3
        for nid in (goal.id, left.id, right.id):
            assert _node(graph, nid).archived_at is None

    def test_unarchive_under_archived_ancestor_refuses_naming_ancestor(
        self, graph: MikadoGraph
    ) -> None:
        goal, left, right = _done_goal_with_two_tasks(graph)
        _mutations.archive_subtree(graph._conn, goal.id)
        with pytest.raises(
            ValueError,
            match=(
                f"Cannot unarchive {right.id} under archived ancestor {goal.id}; "
                "unarchive the ancestor first."
            ),
        ):
            _mutations.unarchive_subtree(graph._conn, right.id)
        for nid in (goal.id, left.id, right.id):
            assert _node(graph, nid).archived_at is not None

    def test_unarchive_deep_node_without_archived_ancestor_clears_only_archived_descendants(
        self, graph: MikadoGraph
    ) -> None:
        root = graph.add_node("live-root")
        mid = graph.add_node("live-mid", parent_id=root.id)
        archived_leaf = graph.add_node("archived-leaf", parent_id=mid.id)
        live_leaf = graph.add_node("live-leaf", parent_id=mid.id)
        _complete(graph, archived_leaf.id)
        _mutations.archive_subtree(graph._conn, archived_leaf.id, now=STAMP)
        assert _mutations.unarchive_subtree(graph._conn, mid.id) == 1
        assert _node(graph, root.id).archived_at is None
        assert _node(graph, mid.id).archived_at is None
        assert _node(graph, archived_leaf.id).archived_at is None
        assert _node(graph, live_leaf.id).archived_at is None

    def test_unarchive_partially_archived_subtree_clears_only_stamped_nodes(
        self, graph: MikadoGraph
    ) -> None:
        goal, left, right = _done_goal_with_two_tasks(graph)
        graph._conn.execute(
            "UPDATE nodes SET archived_at = ? WHERE id IN (?, ?)",
            (STAMP, goal.id, left.id),
        )
        graph._conn.commit()
        assert _mutations.unarchive_subtree(graph._conn, goal.id) == 2
        assert _node(graph, goal.id).archived_at is None
        assert _node(graph, left.id).archived_at is None
        assert _node(graph, right.id).archived_at is None

    def test_unarchive_sibling_subtree_leaves_other_archived_subtrees(
        self, graph: MikadoGraph
    ) -> None:
        root_a = graph.add_node("a")
        child_a = graph.add_node("a-child", parent_id=root_a.id)
        root_b = graph.add_node("b")
        for nid in (child_a.id, root_a.id, root_b.id):
            _complete(graph, nid)
        _mutations.archive_subtree(graph._conn, root_a.id)
        _mutations.archive_subtree(graph._conn, root_b.id)
        assert _mutations.unarchive_subtree(graph._conn, root_a.id) == 2
        assert _node(graph, root_a.id).archived_at is None
        assert _node(graph, child_a.id).archived_at is None
        assert _node(graph, root_b.id).archived_at is not None

    def test_unarchive_non_archived_node_returns_zero(self, graph: MikadoGraph) -> None:
        node = graph.add_node("live")
        assert _mutations.unarchive_subtree(graph._conn, node.id) == 0
        assert _node(graph, node.id).archived_at is None

    def test_unarchive_nested_shelves_from_outer_root_clears_everything(
        self, graph: MikadoGraph
    ) -> None:
        goal, left, right = _done_goal_with_two_tasks(graph)
        grandchild = graph.add_node("grandchild", parent_id=left.id)
        _complete(graph, grandchild.id)
        _mutations.archive_subtree(graph._conn, left.id)
        _mutations.archive_subtree(graph._conn, goal.id)
        # Deepest node of the nested shelf has archived ancestors: refusal.
        with pytest.raises(ValueError, match="under archived ancestor"):
            _mutations.unarchive_subtree(graph._conn, grandchild.id)
        # The archived root owns the shelf; clearing from there restores all.
        assert _mutations.unarchive_subtree(graph._conn, goal.id) == 4
        for nid in (goal.id, left.id, right.id, grandchild.id):
            assert _node(graph, nid).archived_at is None

    def test_unarchive_refuses_under_archived_edge_parent_in_diamond(
        self, graph: MikadoGraph
    ) -> None:
        """Diamond: X's nodes.parent_id is G, but P is an edge-parent too.
        Archiving P stamps X via the edge relation; unarchiving X must refuse
        and name P even though X's parent_id chain (G) is fully live."""
        g = graph.add_node("g")
        x = graph.add_node("x", parent_id=g.id)
        p = graph.add_node("p")
        graph.add_edge(p.id, x.id)
        for nid in (x.id, g.id, p.id):
            _complete(graph, nid)
        assert _mutations.archive_subtree(graph._conn, p.id) == 2
        with pytest.raises(
            ValueError,
            match=(
                f"Cannot unarchive {x.id} under archived ancestor {p.id}; "
                "unarchive the ancestor first."
            ),
        ):
            _mutations.unarchive_subtree(graph._conn, x.id)
        assert _node(graph, p.id).archived_at is not None
        assert _node(graph, x.id).archived_at is not None
        # Restoring from the archived edge-parent clears the shelf.
        assert _mutations.unarchive_subtree(graph._conn, p.id) == 2
        assert _node(graph, p.id).archived_at is None
        assert _node(graph, x.id).archived_at is None

    def test_unarchive_missing_node_raises(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="Node 999 not found"):
            _mutations.unarchive_subtree(graph._conn, 999)


class TestArchivedNodeStatusGuard:
    def test_status_writes_on_archived_node_raise(self, graph: MikadoGraph) -> None:
        node = graph.add_node("done")
        _complete(graph, node.id)
        _mutations.archive_subtree(graph._conn, node.id)
        message = f"Node {node.id} is archived; unarchive it first."
        with pytest.raises(ValueError, match=message):
            graph.mark_pending(node.id)
        with pytest.raises(ValueError, match=message):
            graph.mark_failed(node.id)
        with pytest.raises(ValueError, match=message):
            graph.mark_running(node.id)
        with pytest.raises(ValueError, match=message):
            graph.mark_blocked(node.id)
        with pytest.raises(ValueError, match=message):
            graph.mark_done(node.id)
        assert _node(graph, node.id).status == NodeStatus.DONE

    def test_mark_done_remains_legal_for_non_archived_node(self, graph: MikadoGraph) -> None:
        node = graph.add_node("live")
        graph.mark_running(node.id)
        graph.mark_done(node.id)
        assert _node(graph, node.id).status == NodeStatus.DONE
        assert _node(graph, node.id).archived_at is None

    def test_reads_and_delete_are_not_blocked(self, graph: MikadoGraph) -> None:
        node = graph.add_node("done")
        _complete(graph, node.id)
        _mutations.archive_subtree(graph._conn, node.id)
        assert _node(graph, node.id).archived_at is not None
        assert graph.delete_node(node.id) == 1
        assert graph.get_node(node.id) is None


class TestSetSubtreeStatusSkipsArchived:
    def test_archived_nodes_are_skipped_not_stamped_not_errored(self, graph: MikadoGraph) -> None:
        goal = graph.add_node("goal")
        done_task = graph.add_node("done-task", parent_id=goal.id)
        shelved_task = graph.add_node("shelved-task", parent_id=goal.id)
        _complete(graph, done_task.id)
        _complete(graph, shelved_task.id)
        _mutations.archive_subtree(graph._conn, shelved_task.id, now=STAMP)

        pipeline = StatusPipeline([])
        updated = _status.set_subtree_status(pipeline, graph._conn, goal.id, NodeStatus.DONE)

        assert updated == 1  # only the still-live goal moved
        assert _node(graph, goal.id).status == NodeStatus.DONE
        shelved = _node(graph, shelved_task.id)
        assert shelved.status == NodeStatus.DONE
        assert shelved.archived_at == datetime.fromisoformat(STAMP)

    def test_fully_archived_subtree_stamps_nothing_and_returns_zero(
        self, graph: MikadoGraph
    ) -> None:
        goal, left, right = _done_goal_with_two_tasks(graph)
        _mutations.archive_subtree(graph._conn, goal.id, now=STAMP)
        pipeline = StatusPipeline([])
        assert _status.set_subtree_status(pipeline, graph._conn, goal.id, NodeStatus.BLOCKED) == 0
        for nid in (goal.id, left.id, right.id):
            node = _node(graph, nid)
            assert node.status == NodeStatus.DONE
            assert node.archived_at == datetime.fromisoformat(STAMP)

    def test_cascade_to_done_releases_goal_claim(self, graph: MikadoGraph) -> None:
        """Regression (post-merge F2): a GOAL cascaded to DONE releases its
        goal_claims row, mirroring mark_done/mark_terminal."""
        goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
        graph.add_node("task", parent_id=goal.id)
        assert graph.claim_goal(goal.id, "run-a", now="2026-01-01T00:00:00+00:00")

        pipeline = StatusPipeline([])
        _status.set_subtree_status(pipeline, graph._conn, goal.id, NodeStatus.DONE)

        assert get_goal_claim(graph._conn, goal.id) is None

    def test_cascade_to_pending_clears_worktree_and_run_pins(self, graph: MikadoGraph) -> None:
        """Regression (post-merge F3): resetting a subtree to PENDING clears
        worktree_path/branch_name/run_id exactly as mark_pending does."""
        node = graph.add_node("task")
        graph.mark_running(node.id, worktree_path="/wt/x", branch_name="br/x", run_id="run-1")

        pipeline = StatusPipeline([])
        _status.set_subtree_status(pipeline, graph._conn, node.id, NodeStatus.PENDING)

        node = _node(graph, node.id)
        assert node.status == NodeStatus.PENDING
        assert node.worktree_path is None
        assert node.branch_name is None
        assert node.run_id is None

    def test_missing_root_raises(self, graph: MikadoGraph) -> None:
        pipeline = StatusPipeline([])
        with pytest.raises(ValueError, match="Node 999 not found"):
            _status.set_subtree_status(pipeline, graph._conn, 999, NodeStatus.DONE)

    def test_illegal_transition_mid_subtree_writes_nothing(self, graph: MikadoGraph) -> None:
        """Validate-before-write: an illegal target anywhere in the live set
        aborts the whole cascade — zero nodes change status."""
        goal = graph.add_node("goal")
        pending_task = graph.add_node("pending-task", parent_id=goal.id)
        done_task = graph.add_node("done-task", parent_id=goal.id)
        _complete(graph, done_task.id)
        pipeline = StatusPipeline([])
        # DONE -> PENDING is illegal (VALID_TRANSITIONS[DONE] is empty); the
        # whole subtree must be pre-validated before any write lands.
        with pytest.raises(InvalidTransition):
            _status.set_subtree_status(pipeline, graph._conn, goal.id, NodeStatus.PENDING)
        assert _node(graph, goal.id).status == NodeStatus.PENDING
        assert _node(graph, pending_task.id).status == NodeStatus.PENDING
        assert _node(graph, done_task.id).status == NodeStatus.DONE


class TestAddNodeUnderArchivedParent:
    def test_add_node_rejects_archived_parent(self, graph: MikadoGraph) -> None:
        parent = graph.add_node("parent")
        _complete(graph, parent.id)
        _mutations.archive_subtree(graph._conn, parent.id)
        with pytest.raises(
            ValueError,
            match=f"Cannot add a node under archived parent {parent.id}; unarchive it first.",
        ):
            graph.add_node("child", parent_id=parent.id)
        assert graph.get_children(parent.id) == []

    def test_move_node_rejects_archived_parent(self, graph: MikadoGraph) -> None:
        parent = graph.add_node("parent")
        _complete(graph, parent.id)
        _mutations.archive_subtree(graph._conn, parent.id)
        mover = graph.add_node("mover")
        with pytest.raises(
            ValueError,
            match=f"Cannot add a node under archived parent {parent.id}; unarchive it first.",
        ):
            graph.move_node(mover.id, parent.id)

    def test_add_node_under_archived_descendant_rejected(self, graph: MikadoGraph) -> None:
        parent = graph.add_node("parent")
        child = graph.add_node("child", parent_id=parent.id)
        _complete(graph, child.id)
        _complete(graph, parent.id)
        _mutations.archive_subtree(graph._conn, parent.id)
        with pytest.raises(
            ValueError,
            match=f"Cannot add a node under archived parent {child.id}; unarchive it first.",
        ):
            graph.add_node("grandchild", parent_id=child.id)
        assert graph.get_children(child.id) == []

    def test_add_node_allowed_after_unarchive(self, graph: MikadoGraph) -> None:
        parent = graph.add_node("parent")
        _complete(graph, parent.id)
        _mutations.archive_subtree(graph._conn, parent.id)
        _mutations.unarchive_subtree(graph._conn, parent.id)
        child = graph.add_node("child", parent_id=parent.id)
        assert child.parent_id == parent.id


class TestDeleteNodeHardDeletesArchived:
    def test_delete_subtree_removes_archived_nodes(self, graph: MikadoGraph) -> None:
        goal, left, right = _done_goal_with_two_tasks(graph)
        _mutations.archive_subtree(graph._conn, goal.id)
        assert graph.delete_node(goal.id, cascade=True) == 3
        for nid in (goal.id, left.id, right.id):
            assert graph.get_node(nid) is None


class TestFencedWritesOnArchivedNode:
    """Acceptance: a fenced write repeating an archived node's current status
    is accepted as a reconcile no-op — it must not raise and must not mutate."""

    def test_fenced_writes_on_archived_node_are_safe_noops(self, graph: MikadoGraph) -> None:
        node = graph.add_node("done")
        _complete(graph, node.id)
        _mutations.archive_subtree(graph._conn, node.id, now=STAMP)
        assert graph.mark_terminal(node.id, "stale-run", NodeStatus.DONE) is False
        assert graph.release(node.id, "stale-run") is False
        assert graph.mark_blocked_fenced(node.id, "stale-run") is False
        settled = _node(graph, node.id)
        assert settled.status == NodeStatus.DONE
        assert settled.archived_at == datetime.fromisoformat(STAMP)

    def test_claim_node_on_archived_node_is_a_safe_noop(self, graph: MikadoGraph) -> None:
        """claim_node has no archived guard of its own; it is fenced on
        status IN (pending, failed, blocked), and archived ⇒ DONE, so a claim
        against an archived node lands zero rows and mutates nothing."""
        node = graph.add_node("done")
        _complete(graph, node.id)
        _mutations.archive_subtree(graph._conn, node.id, now=STAMP)
        assert graph.claim_node(node.id, "stale-run", now="2026-07-24T02:00:00+00:00") is False
        settled = _node(graph, node.id)
        assert settled.status == NodeStatus.DONE
        assert settled.run_id is None
        assert settled.archived_at == datetime.fromisoformat(STAMP)


class TestMigrationV3:
    def test_archived_at_persists_across_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "g.db"
        graph = MikadoGraph(db_path)
        node = graph.add_node("done")
        _complete(graph, node.id)
        assert _mutations.archive_subtree(graph._conn, node.id, now=STAMP) == 1
        graph.close()

        reopened = MikadoGraph(db_path)
        try:
            node = _node(reopened, node.id)
            assert node.archived_at == datetime.fromisoformat(STAMP)
            version = reopened._conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == SCHEMA_VERSION
        finally:
            reopened.close()

    def test_migration_v3_adds_column_to_pre_v3_database(self, tmp_path: Path) -> None:
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        from milknado.domains.graph._persistence import create_tables

        create_tables(conn)
        conn.execute("ALTER TABLE nodes DROP COLUMN archived_at")
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()

        graph = MikadoGraph(db_path)  # open must apply v3 and pass validation
        try:
            version = graph._conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == SCHEMA_VERSION
            columns = {row[1] for row in graph._conn.execute("PRAGMA table_info(nodes)")}
            assert "archived_at" in columns
            node = graph.add_node("fresh")
            assert node.archived_at is None
        finally:
            graph.close()
