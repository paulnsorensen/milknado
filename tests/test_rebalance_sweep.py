"""Focused rebalance regression tests."""

from __future__ import annotations

# These tests intentionally exercise private database helpers.
import sqlite3
from typing import cast

from milknado.domains.graph.rebalance import (
    INBOX_DESCRIPTION,
)
from tests.rebalance_helpers import (
    NOW,
    _archived_ids,  # pyright: ignore[reportPrivateUsage]
    _insert_node,  # pyright: ignore[reportPrivateUsage]
    find_archivable_roots,
    group_orphans,
    sweep_archivable,
)

# ── #11: sweep ───────────────────────────────────────────────────────────────


class TestSweep:
    def test_archives_all_done_root_subtree(self, conn: sqlite3.Connection) -> None:
        root = _insert_node(conn, "done goal", "done", kind="goal")
        child = _insert_node(conn, "done task", "done", parent_id=root)
        assert sweep_archivable(conn) == [root]
        assert _archived_ids(conn) == {root, child}

    def test_skips_mixed_status_subtree(self, conn: sqlite3.Connection) -> None:
        root = _insert_node(conn, "goal", "done", kind="goal")
        _ = _insert_node(conn, "pending task", "pending", parent_id=root)
        assert sweep_archivable(conn) == []
        assert _archived_ids(conn) == set()

    def test_skips_pending_root(self, conn: sqlite3.Connection) -> None:
        _ = _insert_node(conn, "pending root", "pending")
        assert sweep_archivable(conn) == []

    def test_skips_already_archived_root(self, conn: sqlite3.Connection) -> None:
        _ = _insert_node(conn, "archived", "done", archived_at=NOW)
        assert sweep_archivable(conn) == []

    def test_claim_does_not_hide_done_subtree_from_sweep(self, conn: sqlite3.Connection) -> None:
        root = _insert_node(conn, "claimed goal", "done", kind="goal")
        _ = conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
            (root, "run-x", 12345, NOW),
        )
        assert sweep_archivable(conn) == [root]
        assert _archived_ids(conn) == {root}

    def test_archives_only_eligible_roots(self, conn: sqlite3.Connection) -> None:
        done_root = _insert_node(conn, "done", "done", kind="goal")
        _ = _insert_node(conn, "live", "in_progress", kind="goal")
        assert sweep_archivable(conn) == [done_root]

    def test_nested_done_subtree_archives_maximal_root_only(
        self, conn: sqlite3.Connection
    ) -> None:
        root = _insert_node(conn, "done root goal", "done", kind="goal")
        child_goal = _insert_node(conn, "done child goal", "done", kind="goal", parent_id=root)
        grandchild = _insert_node(conn, "done leaf", "done", parent_id=child_goal)
        # Exactly one archived root — the maximal one, never children separately.
        assert sweep_archivable(conn) == [root]
        assert _archived_ids(conn) == {root, child_goal, grandchild}

    def test_shared_done_descendant_is_not_selected_as_a_root(
        self, conn: sqlite3.Connection
    ) -> None:
        left = _insert_node(conn, "left", "done", kind="goal")
        right = _insert_node(conn, "right", "done", kind="goal")
        shared = _insert_node(conn, "shared", "done", parent_id=left)
        _ = conn.execute("INSERT INTO edges (parent_id, child_id) VALUES (?, ?)", (right, shared))

        assert find_archivable_roots(conn) == [left, right]
        assert sweep_archivable(conn) == [left, right]
        assert _archived_ids(conn) == {left, right, shared}

    def test_nested_maximal_subtree_under_live_ancestor_is_archived(
        self, conn: sqlite3.Connection
    ) -> None:
        # Acceptance #11: EVERY maximal all-DONE subtree. A done goal under a
        # live roadmap is maximal (its parent is not DONE) and must be swept.
        roadmap = _insert_node(conn, "live roadmap", "in_progress", kind="goal")
        done_goal = _insert_node(conn, "done goal", "done", kind="goal", parent_id=roadmap)
        leaf = _insert_node(conn, "done task", "done", parent_id=done_goal)
        live_sibling = _insert_node(conn, "live task", "pending", parent_id=roadmap)
        assert sweep_archivable(conn) == [done_goal]
        assert _archived_ids(conn) == {done_goal, leaf}
        assert live_sibling not in _archived_ids(conn)
        assert roadmap not in _archived_ids(conn)

    def test_find_archivable_roots_is_read_only(self, conn: sqlite3.Connection) -> None:
        _ = _insert_node(conn, "done goal", "done", kind="goal")
        assert len(find_archivable_roots(conn)) == 1
        assert _archived_ids(conn) == set()  # selection never stamps

    def test_partially_archived_subtree_stamps_only_unarchived(
        self, conn: sqlite3.Connection
    ) -> None:
        stale_stamp = "2020-01-01T00:00:00+00:00"
        root = _insert_node(conn, "done root", "done", kind="goal")
        pre_archived = _insert_node(
            conn, "already archived", "done", parent_id=root, archived_at=stale_stamp
        )
        fresh = _insert_node(conn, "not yet archived", "done", parent_id=root)
        assert sweep_archivable(conn) == [root]
        assert _archived_ids(conn) == {root, pre_archived, fresh}
        # The pre-archived node keeps its original stamp — sweep does not re-stamp it.
        row = cast(
            sqlite3.Row,
            conn.execute("SELECT archived_at FROM nodes WHERE id = ?", (pre_archived,)).fetchone(),
        )
        assert row["archived_at"] == stale_stamp

    def test_claimed_descendant_does_not_block_sweep(self, conn: sqlite3.Connection) -> None:
        root = _insert_node(conn, "done root", "done", kind="goal")
        claimed_child = _insert_node(conn, "done child goal", "done", kind="goal", parent_id=root)
        _ = conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
            (claimed_child, "run-nested", 4242, NOW),
        )
        assert sweep_archivable(conn) == [root]
        assert _archived_ids(conn) == {root, claimed_child}

    def test_claimed_nested_subtree_under_live_ancestor_is_archived(
        self, conn: sqlite3.Connection
    ) -> None:
        roadmap = _insert_node(conn, "live roadmap", "in_progress", kind="goal")
        done_goal = _insert_node(conn, "done goal", "done", kind="goal", parent_id=roadmap)
        _ = conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
            (done_goal, "run-claim", 777, NOW),
        )
        assert sweep_archivable(conn) == [done_goal]
        assert _archived_ids(conn) == {done_goal}


# ── #12: group_orphans ───────────────────────────────────────────────────────


class TestGroupOrphans:
    def test_creates_inbox_and_attaches_orphans(self, conn: sqlite3.Connection) -> None:
        orphan = _insert_node(conn, "bare task", "pending")
        assert group_orphans(conn) == 1
        inbox = cast(
            sqlite3.Row,
            conn.execute(
                "SELECT id, kind, status FROM nodes WHERE description = ?", (INBOX_DESCRIPTION,)
            ).fetchone(),
        )
        assert inbox is not None
        assert (inbox["kind"], inbox["status"]) == ("goal", "pending")
        row = cast(
            sqlite3.Row,
            conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (orphan,)).fetchone(),
        )
        assert row["parent_id"] == inbox["id"]
        edge = cast(
            sqlite3.Row | None,
            conn.execute(
                "SELECT 1 FROM edges WHERE parent_id = ? AND child_id = ?", (inbox["id"], orphan)
            ).fetchone(),
        )
        assert edge is not None

    def test_idempotent_inbox_creation(self, conn: sqlite3.Connection) -> None:
        _ = _insert_node(conn, "orphan a", "pending")
        assert group_orphans(conn) == 1
        assert group_orphans(conn) == 0  # nothing left to move; Inbox reused
        count = cast(
            int,
            conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE description = ?", (INBOX_DESCRIPTION,)
            ).fetchone()[0],
        )
        assert count == 1

    def test_does_not_reattach_parented_nodes(self, conn: sqlite3.Connection) -> None:
        goal = _insert_node(conn, "goal", "in_progress", kind="goal")
        _ = _insert_node(conn, "child", "pending", parent_id=goal)
        assert group_orphans(conn) == 0

    def test_skips_archived_orphans(self, conn: sqlite3.Connection) -> None:
        _ = _insert_node(conn, "archived orphan", "done", archived_at=NOW)
        assert group_orphans(conn) == 0

    def test_skips_claimed_orphan_subtree(self, conn: sqlite3.Connection) -> None:
        goal = _insert_node(conn, "claimed orphan goal", "in_progress", kind="goal")
        _ = conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
            (goal, "run-c", 999, NOW),
        )
        # A task claimed via its own goal claim stays put.
        assert group_orphans(conn) == 0
        row = cast(
            sqlite3.Row,
            conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (goal,)).fetchone(),
        )
        assert row["parent_id"] is None

    def test_inbox_itself_is_never_moved(self, conn: sqlite3.Connection) -> None:
        inbox_id = _insert_node(conn, INBOX_DESCRIPTION, "pending", kind="goal")
        assert group_orphans(conn) == 0  # no orphans -> nothing attaches
        row = cast(
            sqlite3.Row,
            conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (inbox_id,)).fetchone(),
        )
        assert row["parent_id"] is None

    def test_dangling_parent_id_is_not_treated_as_an_orphan(
        self, conn: sqlite3.Connection
    ) -> None:
        parent = _insert_node(conn, "doomed parent", "done", kind="goal")
        child = _insert_node(conn, "task", "pending", parent_id=parent)
        _ = conn.execute("DELETE FROM edges WHERE parent_id = ?", (parent,))
        _ = conn.execute("DELETE FROM nodes WHERE id = ?", (parent,))
        assert group_orphans(conn) == 0
        row = cast(
            sqlite3.Row,
            conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (child,)).fetchone(),
        )
        assert row["parent_id"] == parent

    def test_task_under_archived_parent_is_not_a_root_orphan(
        self, conn: sqlite3.Connection
    ) -> None:
        parent = _insert_node(conn, "archived parent", "done", kind="goal", archived_at=NOW)
        child = _insert_node(conn, "live task under archive", "pending", parent_id=parent)
        assert group_orphans(conn) == 0
        row = cast(
            sqlite3.Row,
            conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (child,)).fetchone(),
        )
        assert row["parent_id"] == parent

    def test_inbox_named_task_does_not_count_as_inbox(self, conn: sqlite3.Connection) -> None:
        # Contract: Inbox identity requires root + GOAL + reserved description.
        # A TASK (or non-root node) sharing the description is not adopted.
        impostor = _insert_node(conn, INBOX_DESCRIPTION, "pending", kind="task")
        orphan = _insert_node(conn, "real orphan", "pending")
        assert group_orphans(conn) == 2  # impostor is itself an orphan task
        goals = cast(
            list[sqlite3.Row],
            conn.execute(
                "SELECT id FROM nodes WHERE description = ? AND kind = 'goal' "
                + "AND parent_id IS NULL",
                (INBOX_DESCRIPTION,),
            ).fetchall(),
        )
        assert len(goals) == 1
        inbox_id = cast(int, goals[0]["id"])
        for node in (impostor, orphan):
            row = cast(
                sqlite3.Row,
                conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (node,)).fetchone(),
            )
            assert row["parent_id"] == inbox_id

    def test_archived_inbox_is_never_reused_for_live_orphans(
        self, conn: sqlite3.Connection
    ) -> None:
        # Acceptance #5: nothing is added under an archived parent. When the
        # existing Inbox has been swept, find-or-create must build a FRESH
        # non-archived Inbox rather than attach live orphans under the shelved
        # one (which would also hide them from default reads).
        archived_inbox = _insert_node(
            conn, INBOX_DESCRIPTION, "done", kind="goal", archived_at=NOW
        )
        orphan = _insert_node(conn, "live orphan", "pending")
        assert group_orphans(conn) == 1
        row = cast(
            sqlite3.Row,
            conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (orphan,)).fetchone(),
        )
        assert row["parent_id"] != archived_inbox
        fresh = cast(
            sqlite3.Row | None,
            conn.execute(
                "SELECT id, archived_at FROM nodes WHERE description = ? AND archived_at IS NULL",
                (INBOX_DESCRIPTION,),
            ).fetchone(),
        )
        assert fresh is not None
        assert row["parent_id"] == fresh["id"]
        # The archived Inbox is never mutated: its stamp and root position stay.
        old = cast(
            sqlite3.Row,
            conn.execute(
                "SELECT archived_at, parent_id FROM nodes WHERE id = ?", (archived_inbox,)
            ).fetchone(),
        )
        assert old["archived_at"] == NOW
        assert old["parent_id"] is None

    def test_zero_orphans_creates_no_inbox(self, conn: sqlite3.Connection) -> None:
        # Contract: the Inbox is find-or-created only when an orphan attaches.
        goal = _insert_node(conn, "live goal", "in_progress", kind="goal")
        _ = _insert_node(conn, "parented child", "pending", parent_id=goal)
        assert group_orphans(conn) == 0
        inbox = cast(
            sqlite3.Row | None,
            conn.execute(
                "SELECT 1 FROM nodes WHERE description = ?", (INBOX_DESCRIPTION,)
            ).fetchone(),
        )
        assert inbox is None
