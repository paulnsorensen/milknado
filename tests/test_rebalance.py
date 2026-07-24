"""Tests for the rebalance engine — acceptance #11-#15.

Domain functions run against in-memory sqlite + create_tables; the app layer
runs against a tmp_path project root with a fake GitPort at the boundary.
"""

from __future__ import annotations

import sqlite3
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest

import milknado.app.rebalance as app_rebalance
from milknado.adapters.git import GitAdapter
from milknado.app.rebalance import rebalance
from milknado.domains.common.errors import GitOperationError
from milknado.domains.graph._persistence import create_tables
from milknado.domains.graph.rebalance import (
    INBOX_DESCRIPTION,
    RebalanceReport,
    find_archivable_roots,
    group_orphans,
    open_db,
    render_report,
    structural_report,
    sweep_archivable,
)

NOW = "2026-07-24T00:00:00+00:00"


@pytest.fixture()
def conn() -> Generator[sqlite3.Connection, None, None]:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    create_tables(c)
    yield c
    c.close()


def _insert_node(
    conn: sqlite3.Connection,
    description: str,
    status: str,
    *,
    kind: str = "task",
    parent_id: int | None = None,
    archived_at: str | None = None,
    worktree_path: str | None = None,
    branch_name: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO nodes (description, status, parent_id, kind, created_at, archived_at, "
        "worktree_path, branch_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (description, status, parent_id, kind, NOW, archived_at, worktree_path, branch_name),
    )
    node_id = cur.lastrowid
    assert node_id is not None
    if parent_id is not None:
        conn.execute("INSERT INTO edges (parent_id, child_id) VALUES (?, ?)", (parent_id, node_id))
    return node_id


def _register_run(conn: sqlite3.Connection, node_id: int, status: str = "completed") -> None:
    conn.execute(
        "INSERT INTO runs (run_id, node_id, status, log_path, started_at) VALUES (?, ?, ?, '', ?)",
        (f"run-{node_id}", node_id, status, NOW),
    )


def _archived_ids(conn: sqlite3.Connection) -> set[int]:
    return {r[0] for r in conn.execute("SELECT id FROM nodes WHERE archived_at IS NOT NULL")}


class FakeGit:
    """Records teardown calls; configured failures raise like GitAdapter does."""

    def __init__(self) -> None:
        self.removed: list[Path] = []
        self.deleted_branches: list[str] = []
        self.pruned = 0
        self.fail_remove: set[str] = set()
        self.fail_delete: set[str] = set()

    def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
        if str(path) in self.fail_remove:
            raise GitOperationError("worktree remove", "simulated teardown failure")
        self.removed.append(path)

    def delete_branch(self, branch: str) -> None:
        if branch in self.fail_delete:
            raise GitOperationError("branch -d", "branch not fully merged")
        self.deleted_branches.append(branch)

    def prune_worktrees(self) -> None:
        self.pruned += 1

    def worktree_teardown_blocker(self, path: Path, target: str = "HEAD") -> str | None:
        if str(path) in self.fail_remove:
            return "simulated teardown failure"
        return None


# ── #11: sweep ───────────────────────────────────────────────────────────────


class TestSweep:
    def test_archives_all_done_root_subtree(self, conn: sqlite3.Connection) -> None:
        root = _insert_node(conn, "done goal", "done", kind="goal")
        child = _insert_node(conn, "done task", "done", parent_id=root)
        assert sweep_archivable(conn) == [root]
        assert _archived_ids(conn) == {root, child}

    def test_skips_mixed_status_subtree(self, conn: sqlite3.Connection) -> None:
        root = _insert_node(conn, "goal", "done", kind="goal")
        _insert_node(conn, "pending task", "pending", parent_id=root)
        assert sweep_archivable(conn) == []
        assert _archived_ids(conn) == set()

    def test_skips_pending_root(self, conn: sqlite3.Connection) -> None:
        _insert_node(conn, "pending root", "pending")
        assert sweep_archivable(conn) == []

    def test_skips_already_archived_root(self, conn: sqlite3.Connection) -> None:
        _insert_node(conn, "archived", "done", archived_at=NOW)
        assert sweep_archivable(conn) == []

    def test_skips_claimed_subtree(self, conn: sqlite3.Connection) -> None:
        root = _insert_node(conn, "claimed goal", "done", kind="goal")
        conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
            (root, "run-x", 12345, NOW),
        )
        assert sweep_archivable(conn) == []
        assert _archived_ids(conn) == set()

    def test_archives_only_eligible_roots(self, conn: sqlite3.Connection) -> None:
        done_root = _insert_node(conn, "done", "done", kind="goal")
        _insert_node(conn, "live", "in_progress", kind="goal")
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
        _insert_node(conn, "done goal", "done", kind="goal")
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
        row = conn.execute(
            "SELECT archived_at FROM nodes WHERE id = ?", (pre_archived,)
        ).fetchone()
        assert row["archived_at"] == stale_stamp

    def test_claimed_descendant_blocks_sweep(self, conn: sqlite3.Connection) -> None:
        root = _insert_node(conn, "done root", "done", kind="goal")
        claimed_child = _insert_node(conn, "done child goal", "done", kind="goal", parent_id=root)
        conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
            (claimed_child, "run-nested", 4242, NOW),
        )
        assert sweep_archivable(conn) == []
        assert _archived_ids(conn) == set()

    def test_claimed_nested_subtree_under_live_ancestor_is_skipped(
        self, conn: sqlite3.Connection
    ) -> None:
        roadmap = _insert_node(conn, "live roadmap", "in_progress", kind="goal")
        done_goal = _insert_node(conn, "done goal", "done", kind="goal", parent_id=roadmap)
        conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
            (done_goal, "run-claim", 777, NOW),
        )
        assert sweep_archivable(conn) == []
        assert _archived_ids(conn) == set()


# ── #12: group_orphans ───────────────────────────────────────────────────────


class TestGroupOrphans:
    def test_creates_inbox_and_attaches_orphans(self, conn: sqlite3.Connection) -> None:
        orphan = _insert_node(conn, "bare task", "pending")
        assert group_orphans(conn) == 1
        inbox = conn.execute(
            "SELECT id, kind, status FROM nodes WHERE description = ?",
            (INBOX_DESCRIPTION,),
        ).fetchone()
        assert inbox is not None
        assert (inbox["kind"], inbox["status"]) == ("goal", "pending")
        row = conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (orphan,)).fetchone()
        assert row["parent_id"] == inbox["id"]
        edge = conn.execute(
            "SELECT 1 FROM edges WHERE parent_id = ? AND child_id = ?",
            (inbox["id"], orphan),
        ).fetchone()
        assert edge is not None

    def test_idempotent_inbox_creation(self, conn: sqlite3.Connection) -> None:
        _insert_node(conn, "orphan a", "pending")
        assert group_orphans(conn) == 1
        assert group_orphans(conn) == 0  # nothing left to move; Inbox reused
        count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE description = ?", (INBOX_DESCRIPTION,)
        ).fetchone()[0]
        assert count == 1

    def test_does_not_reattach_parented_nodes(self, conn: sqlite3.Connection) -> None:
        goal = _insert_node(conn, "goal", "in_progress", kind="goal")
        _insert_node(conn, "child", "pending", parent_id=goal)
        assert group_orphans(conn) == 0

    def test_skips_archived_orphans(self, conn: sqlite3.Connection) -> None:
        _insert_node(conn, "archived orphan", "done", archived_at=NOW)
        assert group_orphans(conn) == 0

    def test_skips_claimed_orphan_subtree(self, conn: sqlite3.Connection) -> None:
        goal = _insert_node(conn, "claimed orphan goal", "in_progress", kind="goal")
        conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
            (goal, "run-c", 999, NOW),
        )
        # A task claimed via its own goal claim stays put.
        assert group_orphans(conn) == 0
        row = conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (goal,)).fetchone()
        assert row["parent_id"] is None

    def test_inbox_itself_is_never_moved(self, conn: sqlite3.Connection) -> None:
        inbox_id = _insert_node(conn, INBOX_DESCRIPTION, "pending", kind="goal")
        assert group_orphans(conn) == 0  # no orphans -> nothing attaches
        row = conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (inbox_id,)).fetchone()
        assert row["parent_id"] is None

    def test_dangling_parent_orphan_is_attached(self, conn: sqlite3.Connection) -> None:
        parent = _insert_node(conn, "doomed parent", "done", kind="goal")
        child = _insert_node(conn, "task", "pending", parent_id=parent)
        conn.execute("DELETE FROM edges WHERE parent_id = ?", (parent,))
        conn.execute("DELETE FROM nodes WHERE id = ?", (parent,))
        assert group_orphans(conn) == 1
        row = conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (child,)).fetchone()
        inbox = conn.execute(
            "SELECT id FROM nodes WHERE description = ?", (INBOX_DESCRIPTION,)
        ).fetchone()
        assert row["parent_id"] == inbox["id"]

    def test_task_under_archived_parent_is_orphaned(self, conn: sqlite3.Connection) -> None:
        parent = _insert_node(conn, "archived parent", "done", kind="goal", archived_at=NOW)
        child = _insert_node(conn, "live task under archive", "pending", parent_id=parent)
        assert group_orphans(conn) == 1
        row = conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (child,)).fetchone()
        inbox = conn.execute(
            "SELECT id FROM nodes WHERE description = ?", (INBOX_DESCRIPTION,)
        ).fetchone()
        assert row["parent_id"] == inbox["id"]

    def test_inbox_named_task_does_not_count_as_inbox(self, conn: sqlite3.Connection) -> None:
        # Contract: Inbox identity requires root + GOAL + reserved description.
        # A TASK (or non-root node) sharing the description is not adopted.
        impostor = _insert_node(conn, INBOX_DESCRIPTION, "pending", kind="task")
        orphan = _insert_node(conn, "real orphan", "pending")
        assert group_orphans(conn) == 2  # impostor is itself an orphan task
        goals = conn.execute(
            "SELECT id FROM nodes WHERE description = ? AND kind = 'goal' AND parent_id IS NULL",
            (INBOX_DESCRIPTION,),
        ).fetchall()
        assert len(goals) == 1
        inbox_id = goals[0]["id"]
        for node in (impostor, orphan):
            row = conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (node,)).fetchone()
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
        row = conn.execute("SELECT parent_id FROM nodes WHERE id = ?", (orphan,)).fetchone()
        assert row["parent_id"] != archived_inbox
        fresh = conn.execute(
            "SELECT id, archived_at FROM nodes WHERE description = ? AND archived_at IS NULL",
            (INBOX_DESCRIPTION,),
        ).fetchone()
        assert fresh is not None
        assert row["parent_id"] == fresh["id"]
        # The archived Inbox is never mutated: its stamp and root position stay.
        old = conn.execute(
            "SELECT archived_at, parent_id FROM nodes WHERE id = ?", (archived_inbox,)
        ).fetchone()
        assert old["archived_at"] == NOW
        assert old["parent_id"] is None

    def test_zero_orphans_creates_no_inbox(self, conn: sqlite3.Connection) -> None:
        # Contract: the Inbox is find-or-created only when an orphan attaches.
        goal = _insert_node(conn, "live goal", "in_progress", kind="goal")
        _insert_node(conn, "parented child", "pending", parent_id=goal)
        assert group_orphans(conn) == 0
        inbox = conn.execute(
            "SELECT 1 FROM nodes WHERE description = ?", (INBOX_DESCRIPTION,)
        ).fetchone()
        assert inbox is None


# ── #13: structural report (report-only) ─────────────────────────────────────


class TestStructuralReport:
    def _goal_chain(self, conn: sqlite3.Connection, depth: int) -> list[int]:
        ids: list[int] = []
        parent: int | None = None
        for i in range(depth):
            node_id = _insert_node(
                conn, f"chain goal {i}", "in_progress", kind="goal", parent_id=parent
            )
            ids.append(node_id)
            parent = node_id
        return ids

    def test_chain_at_depth_4_is_reported(self, conn: sqlite3.Connection) -> None:
        ids = self._goal_chain(conn, 4)
        report = structural_report(conn)
        assert report.chains == (tuple(ids),)
        assert report.lopsided == ()

    def test_chain_at_depth_3_is_not_reported(self, conn: sqlite3.Connection) -> None:
        self._goal_chain(conn, 3)
        assert structural_report(conn).chains == ()

    def test_chain_head_only_is_reported_for_longer_chains(self, conn: sqlite3.Connection) -> None:
        ids = self._goal_chain(conn, 5)
        report = structural_report(conn)
        # One maximal chain, not overlapping sub-chains at depths 4 and 5.
        assert report.chains == (tuple(ids),)

    def test_branching_breaks_the_chain(self, conn: sqlite3.Connection) -> None:
        g1 = _insert_node(conn, "g1", "in_progress", kind="goal")
        g2 = _insert_node(conn, "g2", "in_progress", kind="goal", parent_id=g1)
        g3 = _insert_node(conn, "g3", "in_progress", kind="goal", parent_id=g2)
        # g3 forks into two goal children, so no single-child run reaches 4.
        _insert_node(conn, "g4a", "in_progress", kind="goal", parent_id=g3)
        _insert_node(conn, "g4b", "in_progress", kind="goal", parent_id=g3)
        assert structural_report(conn).chains == ()

    def test_goal_with_20_direct_tasks_is_reported(self, conn: sqlite3.Connection) -> None:
        goal = _insert_node(conn, "fat goal", "in_progress", kind="goal")
        for i in range(20):
            _insert_node(conn, f"task {i}", "pending", parent_id=goal)
        assert structural_report(conn).lopsided == (goal,)

    def test_goal_with_19_direct_tasks_is_not_reported(self, conn: sqlite3.Connection) -> None:
        goal = _insert_node(conn, "almost fat goal", "in_progress", kind="goal")
        for i in range(19):
            _insert_node(conn, f"task {i}", "pending", parent_id=goal)
        assert structural_report(conn).lopsided == ()

    def test_archived_nodes_are_excluded(self, conn: sqlite3.Connection) -> None:
        parent: int | None = None
        for i in range(4):
            parent = _insert_node(
                conn,
                f"archived chain goal {i}",
                "done",
                kind="goal",
                parent_id=parent,
                archived_at=NOW,
            )
        assert structural_report(conn).chains == ()

    def test_structural_report_never_mutates(self, conn: sqlite3.Connection) -> None:
        self._goal_chain(conn, 4)
        goal = _insert_node(conn, "fat goal", "in_progress", kind="goal")
        for i in range(20):
            _insert_node(conn, f"task {i}", "pending", parent_id=goal)
        nodes_before = conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        edges_before = conn.execute("SELECT * FROM edges ORDER BY parent_id, child_id").fetchall()
        report = structural_report(conn)
        assert report.chains and report.lopsided  # findings exist, so mutation would show
        assert conn.execute("SELECT * FROM nodes ORDER BY id").fetchall() == nodes_before
        assert (
            conn.execute("SELECT * FROM edges ORDER BY parent_id, child_id").fetchall()
            == edges_before
        )


# ── #14: reap ────────────────────────────────────────────────────────────────


def _project_with_db(tmp_path: Path) -> sqlite3.Connection:
    return open_db(tmp_path / ".milknado" / "milknado.db")


class TestReap:
    def test_reap_removes_worktrees_and_deletes_branches(self, tmp_path: Path) -> None:
        conn = _project_with_db(tmp_path)
        node = _insert_node(
            conn,
            "archived with worktree",
            "done",
            archived_at=NOW,
            worktree_path=str(tmp_path / ".worktrees" / "wt-1"),
            branch_name="milknado/1-wt",
        )
        _register_run(conn, node)
        # No registered run -> skipped silently (not milknado-managed).
        _insert_node(
            conn,
            "archived stray",
            "done",
            archived_at=NOW,
            worktree_path=str(tmp_path / "elsewhere"),
            branch_name="stray",
        )
        conn.commit()
        conn.close()

        git = FakeGit()
        report = rebalance(tmp_path, sweep=False, restructure=False, git=git)

        assert git.removed == [tmp_path / ".worktrees" / "wt-1"]
        assert git.deleted_branches == ["milknado/1-wt"]
        assert git.pruned == 1
        assert report.reaped == (str(tmp_path / ".worktrees" / "wt-1"),)
        assert report.branches_deleted == ("milknado/1-wt",)
        assert report.preserved == ()

    def test_done_but_unarchived_worktree_is_never_reaped(self, tmp_path: Path) -> None:
        # Archive gates destruction: a DONE node whose subtree was never
        # archived keeps its worktree, even with terminal-run evidence.
        conn = _project_with_db(tmp_path)
        node = _insert_node(
            conn,
            "done but live",
            "done",
            worktree_path=str(tmp_path / ".worktrees" / "wt-live"),
            branch_name="milknado/live",
        )
        _register_run(conn, node)
        conn.commit()
        conn.close()

        git = FakeGit()
        report = rebalance(tmp_path, sweep=False, restructure=False, git=git)
        assert git.removed == [] and git.deleted_branches == [] and git.pruned == 1
        assert report.reaped == () and report.branches_deleted == () and report.preserved == ()

    def test_teardown_failure_is_preserved_and_pass_continues(self, tmp_path: Path) -> None:
        conn = _project_with_db(tmp_path)
        bad_wt = str(tmp_path / ".worktrees" / "wt-bad")
        good_wt = str(tmp_path / ".worktrees" / "wt-good")
        bad = _insert_node(
            conn,
            "bad",
            "done",
            archived_at=NOW,
            worktree_path=bad_wt,
            branch_name="milknado/bad",
        )
        _register_run(conn, bad)
        good = _insert_node(
            conn,
            "good",
            "done",
            archived_at=NOW,
            worktree_path=good_wt,
            branch_name="milknado/good",
        )
        _register_run(conn, good)
        conn.commit()
        conn.close()

        git = FakeGit()
        git.fail_remove.add(bad_wt)
        report = rebalance(tmp_path, sweep=False, restructure=False, git=git)

        # Fail closed: preserved, never forced, branch kept, pass continues,
        # prune still runs.
        assert report.preserved == (bad_wt,)
        assert report.reaped == (good_wt,)
        assert git.removed == [Path(good_wt)]
        assert git.deleted_branches == ["milknado/good"]
        assert git.pruned == 1

    def test_stale_worktree_path_is_preserved_and_prune_still_runs(self, tmp_path: Path) -> None:
        # A worktree removed externally must not brick the pass: the real
        # adapter's remove_worktree raises, the node is preserved, and the
        # prune that heals the stale registration still runs.
        repo = tmp_path
        _git(repo, "init", "-b", "main")
        _git(repo, "commit", "--allow-empty", "-m", "init")
        conn = _project_with_db(tmp_path)
        gone = str(tmp_path / ".worktrees" / "wt-gone")
        node = _insert_node(
            conn,
            "stale",
            "done",
            archived_at=NOW,
            worktree_path=gone,
            branch_name=None,
        )
        _register_run(conn, node)
        conn.commit()
        conn.close()

        report = rebalance(tmp_path, sweep=False, restructure=False, git=GitAdapter(repo))
        assert report.preserved == (gone,)
        assert report.reaped == ()

    def test_unmerged_branch_is_kept_and_reported_worktree_stays_reaped(
        self, tmp_path: Path
    ) -> None:
        conn = _project_with_db(tmp_path)
        wt = str(tmp_path / ".worktrees" / "wt-2")
        node = _insert_node(
            conn,
            "archived",
            "done",
            archived_at=NOW,
            worktree_path=wt,
            branch_name="milknado/2-y",
        )
        _register_run(conn, node)
        conn.commit()
        conn.close()

        git = FakeGit()
        git.fail_delete.add("milknado/2-y")
        report = rebalance(tmp_path, sweep=False, restructure=False, git=git)
        # Single-bucket membership: the worktree was removed cleanly, so it is
        # reaped ONLY — never also listed as preserved (it no longer exists).
        assert report.reaped == (wt,)
        assert report.preserved == ()
        assert report.branches_deleted == ()
        # The surviving artifact is the branch; it is surfaced by name.
        assert report.branches_kept == ("milknado/2-y",)
        rendered = render_report(report)
        assert "milknado/2-y" in rendered
        assert "unmerged" in rendered


# ── #15: dry-run ─────────────────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_mutates_nothing_and_prefixes_report(self, tmp_path: Path) -> None:
        conn = _project_with_db(tmp_path)
        done_root = _insert_node(conn, "done goal", "done", kind="goal")
        orphan = _insert_node(conn, "orphan", "pending")
        wt = str(tmp_path / ".worktrees" / "wt-3")
        reaped_node = _insert_node(
            conn, "done wt", "done", worktree_path=wt, branch_name="milknado/3-z"
        )
        _register_run(conn, reaped_node)
        conn.commit()
        conn.close()

        git = FakeGit()
        report = rebalance(tmp_path, dry_run=True, git=git)

        # Counts reflect the plan (both all-DONE roots are swept; the reaped
        # node's worktree is planned via the would-be-archived gate)...
        assert report.archived_roots == (done_root, reaped_node)
        assert report.moved_tasks == 1
        assert report.reaped == (wt,)
        # ...but no branch deletion is promised: the merged check never runs.
        assert report.branches_deleted == ()
        # ...and nothing happened: no git calls, no stamps, no moves, no Inbox.
        assert git.removed == [] and git.deleted_branches == [] and git.pruned == 0

        check = sqlite3.connect(str(tmp_path / ".milknado" / "milknado.db"))
        check.row_factory = sqlite3.Row
        assert _archived_ids(check) == set()
        row = check.execute("SELECT parent_id FROM nodes WHERE id = ?", (orphan,)).fetchone()
        assert row["parent_id"] is None
        inbox = check.execute(
            "SELECT 1 FROM nodes WHERE description = ?", (INBOX_DESCRIPTION,)
        ).fetchone()
        assert inbox is None
        check.close()

        rendered = render_report(report, dry_run=True)
        lines = rendered.splitlines()
        assert len(lines) == 5
        assert all(line.startswith("[dry-run] Would ") for line in lines)

    def test_dry_run_never_opens_a_write_transaction(self, tmp_path: Path) -> None:
        # A concurrent writer holds the RESERVED lock; a read-only dry-run must
        # still compute the full plan (the old BEGIN IMMEDIATE path would fail
        # with "database is locked" here).
        conn = _project_with_db(tmp_path)
        done_root = _insert_node(conn, "done goal", "done", kind="goal")
        _insert_node(conn, "orphan", "pending")
        conn.commit()
        conn.close()

        holder = sqlite3.connect(str(tmp_path / ".milknado" / "milknado.db"))
        holder.execute("BEGIN IMMEDIATE")
        holder.execute(
            "INSERT INTO nodes (description, status, kind, created_at) VALUES (?, ?, ?, ?)",
            ("held", "pending", "task", NOW),
        )
        try:
            report = rebalance(tmp_path, dry_run=True, git=FakeGit())
        finally:
            holder.rollback()
            holder.close()
        assert report.archived_roots == (done_root,)
        assert report.moved_tasks == 1

    def test_dry_run_on_missing_project_creates_no_database(self, tmp_path: Path) -> None:
        report = rebalance(tmp_path, dry_run=True, git=FakeGit())
        assert report == RebalanceReport()
        assert not (tmp_path / ".milknado" / "milknado.db").exists()

    def test_dry_run_plan_renders_new_inbox_when_one_would_be_created(
        self, tmp_path: Path
    ) -> None:
        conn = _project_with_db(tmp_path)
        _insert_node(conn, "orphan", "pending")
        conn.commit()
        conn.close()

        report = rebalance(tmp_path, dry_run=True, git=FakeGit())
        assert report.inbox_id is None  # not created in a dry-run...
        assert report.moved_tasks == 1
        # ...but the plan says what the apply will do: attach under a new Inbox.
        assert "under Inbox (new)" in render_report(report, dry_run=True)


# ── #16: report ordering and alignment ───────────────────────────────────────


class TestRenderReport:
    def _report(self) -> RebalanceReport:
        return RebalanceReport(
            archived_roots=(1, 2, 3),
            inbox_id=9,
            moved_tasks=12,
            chains=((4, 5, 6, 7),),
            lopsided=(8,),
            reaped=("/a", "/b"),
            branches_deleted=("b1",),
            preserved=(),
        )

    def test_ordering_sweep_restructure_reap(self) -> None:
        lines = render_report(self._report()).splitlines()
        assert lines[0].startswith("Sweep:")
        assert all(line.startswith("Restructure:") for line in lines[1:4])
        assert lines[4].startswith("Reap:")

    def test_counters_equal_width(self) -> None:
        rendered = render_report(self._report())
        # Widest counter is moved_tasks=12, so every counter is zero-padded to 2.
        assert "archived 03 subtree(s)" in rendered
        assert "moved 12 orphan task(s)" in rendered
        assert "found 01 deep chain(s)" in rendered
        assert "found 01 lopsided goal(s)" in rendered
        assert "removed 02 worktree(s)" in rendered
        assert "deleted 01 branch(es)" in rendered
        assert "preserved 00" in rendered

    def test_dry_run_preserves_alignment(self) -> None:
        report = self._report()
        plain = render_report(report).splitlines()
        dry = render_report(report, dry_run=True).splitlines()
        prefix = "[dry-run] Would "
        assert len(plain) == len(dry)
        for p, d in zip(plain, dry, strict=True):
            assert d == prefix + p

    def test_exact_line_strings(self) -> None:
        assert render_report(self._report()).splitlines() == [
            "Sweep: archived 03 subtree(s) (roots: 1, 2, 3)",
            "Restructure: moved 12 orphan task(s) under Inbox (id 9)",
            "Restructure: found 01 deep chain(s) (roots: 4)",
            "Restructure: found 01 lopsided goal(s) (ids: 8)",
            "Reap: removed 02 worktree(s), deleted 01 branch(es), preserved 00",
        ]

    def test_dry_run_exact_line_strings(self) -> None:
        assert render_report(self._report(), dry_run=True).splitlines() == [
            "[dry-run] Would Sweep: archived 03 subtree(s) (roots: 1, 2, 3)",
            "[dry-run] Would Restructure: moved 12 orphan task(s) under Inbox (id 9)",
            "[dry-run] Would Restructure: found 01 deep chain(s) (roots: 4)",
            "[dry-run] Would Restructure: found 01 lopsided goal(s) (ids: 8)",
            "[dry-run] Would Reap: removed 02 worktree(s), deleted 01 branch(es), preserved 00",
        ]

    def test_kept_branches_render_suffix(self) -> None:
        report = RebalanceReport(reaped=("/a",), branches_kept=("b1", "b2"))
        line = render_report(report).splitlines()[4]
        assert line == (
            "Reap: removed 1 worktree(s), deleted 0 branch(es), preserved 0, "
            "kept 2 branch(es) (unmerged: b1, b2)"
        )

    def test_dry_run_renders_would_be_inbox_as_new(self) -> None:
        report = RebalanceReport(moved_tasks=2, inbox_id=None)
        line = render_report(report, dry_run=True).splitlines()[1]
        assert line == "[dry-run] Would Restructure: moved 2 orphan task(s) under Inbox (new)"
        # A real run with no Inbox and nothing moved still says "no Inbox".
        assert "under Inbox (no Inbox)" in render_report(RebalanceReport()).splitlines()[1]

    def test_empty_report_exact_strings(self) -> None:
        assert render_report(RebalanceReport()).splitlines() == [
            "Sweep: archived 0 subtree(s) (roots: none)",
            "Restructure: moved 0 orphan task(s) under Inbox (no Inbox)",
            "Restructure: found 0 deep chain(s) (roots: none)",
            "Restructure: found 0 lopsided goal(s) (ids: none)",
            "Reap: removed 0 worktree(s), deleted 0 branch(es), preserved 0",
        ]


# ── #14: reap skip edges ─────────────────────────────────────────────────────


class TestReapSkips:
    def test_running_run_is_not_terminal_evidence(self, tmp_path: Path) -> None:
        conn = _project_with_db(tmp_path)
        node = _insert_node(
            conn,
            "archived but still running",
            "done",
            archived_at=NOW,
            worktree_path=str(tmp_path / ".worktrees" / "wt-run"),
            branch_name="milknado/9-run",
        )
        _register_run(conn, node, status="running")
        conn.commit()
        conn.close()

        git = FakeGit()
        report = rebalance(tmp_path, sweep=False, restructure=False, git=git)
        assert git.removed == [] and git.deleted_branches == [] and git.pruned == 1
        assert report.reaped == () and report.branches_deleted == () and report.preserved == ()

    def test_archived_node_without_worktree_is_skipped(self, tmp_path: Path) -> None:
        conn = _project_with_db(tmp_path)
        node = _insert_node(
            conn, "archived no worktree", "done", archived_at=NOW, branch_name="milknado/9-none"
        )
        _register_run(conn, node)
        conn.commit()
        conn.close()

        git = FakeGit()
        report = rebalance(tmp_path, sweep=False, restructure=False, git=git)
        assert git.removed == [] and git.deleted_branches == []
        assert report.reaped == () and report.branches_deleted == () and report.preserved == ()

    def test_dry_run_reap_does_not_prune(self, tmp_path: Path) -> None:
        conn = _project_with_db(tmp_path)
        node = _insert_node(
            conn,
            "archived wt",
            "done",
            archived_at=NOW,
            worktree_path=str(tmp_path / ".worktrees" / "wt-dry"),
            branch_name="milknado/9-dry",
        )
        _register_run(conn, node)
        conn.commit()
        conn.close()

        git = FakeGit()
        rebalance(tmp_path, dry_run=True, sweep=False, restructure=False, git=git)
        assert git.pruned == 0


# ── #15: dry-run counters match the real run ──────────────────────────────────


def _seed_rebalance_project(root: Path) -> dict[str, int | str]:
    conn = _project_with_db(root)
    done_root = _insert_node(conn, "done goal", "done", kind="goal")
    _insert_node(conn, "orphan", "pending")
    wt = str(root / ".worktrees" / "wt-x")
    reaped = _insert_node(conn, "done wt", "done", worktree_path=wt, branch_name="milknado/x")
    _register_run(conn, reaped)
    conn.commit()
    conn.close()
    return {"done_root": done_root, "wt": wt}


class TestDryRunParity:
    def test_dry_run_counters_equal_real_run_counters(self, tmp_path: Path) -> None:
        dry_root = tmp_path / "dry"
        dry_root.mkdir()
        real_root = tmp_path / "real"
        real_root.mkdir()
        _seed_rebalance_project(dry_root)
        _seed_rebalance_project(real_root)

        dry = rebalance(dry_root, dry_run=True, git=FakeGit())
        real = rebalance(real_root, git=FakeGit())

        def counts(r: RebalanceReport) -> tuple[int, int, int, int]:
            return (
                len(r.archived_roots),
                r.moved_tasks,
                len(r.reaped),
                len(r.preserved),
            )

        assert counts(dry) == counts(real) == (2, 1, 1, 0)
        # Branch deletions are the deliberate plan/apply divergence: the dry
        # plan never promises them (the merged check only runs for real).
        assert dry.branches_deleted == ()
        assert len(real.branches_deleted) == 1
        assert real.inbox_id is not None


# ── #12/#5: an archived Inbox never attracts live orphans (end to end) ───────


class TestArchivedInboxEndToEnd:
    def test_orphan_attaches_to_fresh_inbox_after_sweep_shelves_the_old_one(
        self, tmp_path: Path
    ) -> None:
        # The reproduced defect: Inbox fills with DONE work -> sweep archives
        # the Inbox subtree -> a new orphan must attach to a FRESH Inbox, not
        # the archived one.
        conn = _project_with_db(tmp_path)
        inbox = _insert_node(conn, INBOX_DESCRIPTION, "done", kind="goal")
        _insert_node(conn, "finished inbox task", "done", parent_id=inbox)
        orphan = _insert_node(conn, "fresh live orphan", "pending")
        conn.commit()
        conn.close()

        report = rebalance(tmp_path, git=FakeGit())
        assert inbox in report.archived_roots  # sweep shelved the old Inbox
        assert report.moved_tasks == 1
        assert report.inbox_id is not None and report.inbox_id != inbox

        check = sqlite3.connect(str(tmp_path / ".milknado" / "milknado.db"))
        check.row_factory = sqlite3.Row
        row = check.execute("SELECT parent_id FROM nodes WHERE id = ?", (orphan,)).fetchone()
        assert row["parent_id"] == report.inbox_id
        parent = check.execute(
            "SELECT archived_at, description FROM nodes WHERE id = ?", (report.inbox_id,)
        ).fetchone()
        assert parent["archived_at"] is None
        assert parent["description"] == INBOX_DESCRIPTION
        # The shelved Inbox was never mutated.
        old = check.execute("SELECT archived_at FROM nodes WHERE id = ?", (inbox,)).fetchone()
        assert old["archived_at"] is not None
        check.close()


# ── #12/#5: dry-run predicts the fresh Inbox when the old one is swept ───────


def _seed_swept_inbox_project(root: Path) -> int:
    """An all-DONE Inbox subtree (sweep-eligible) plus one live orphan."""
    conn = _project_with_db(root)
    inbox = _insert_node(conn, INBOX_DESCRIPTION, "done", kind="goal")
    _insert_node(conn, "finished inbox task", "done", parent_id=inbox)
    _insert_node(conn, "fresh live orphan", "pending")
    conn.commit()
    conn.close()
    return inbox


class TestDryRunSweptInboxParity:
    def test_dry_run_predicts_new_inbox_and_matches_real_apply(self, tmp_path: Path) -> None:
        dry_root = tmp_path / "dry"
        dry_root.mkdir()
        real_root = tmp_path / "real"
        real_root.mkdir()
        dry_inbox = _seed_swept_inbox_project(dry_root)
        real_inbox = _seed_swept_inbox_project(real_root)

        dry = rebalance(dry_root, dry_run=True, git=FakeGit())
        real = rebalance(real_root, git=FakeGit())

        # The dry plan predicts a fresh Inbox instead of naming the doomed id.
        assert dry.inbox_id is None
        plan = render_report(dry, dry_run=True)
        assert "under Inbox (new)" in plan
        assert f"under Inbox (id {dry_inbox})" not in plan

        # Parity: the real apply archived the old Inbox, moved the same orphan
        # under a FRESH Inbox, exactly as the dry plan predicted.
        assert dry.archived_roots == real.archived_roots == (real_inbox,)
        assert dry.moved_tasks == real.moved_tasks == 1
        assert real.inbox_id is not None and real.inbox_id != real_inbox


# ── #15: dry-run structural findings match the real run ──────────────────────


def _seed_chain_project(root: Path) -> tuple[int, ...]:
    """A DONE depth-5 chain (swept away) plus a LIVE depth-4 chain (reported)."""
    conn = _project_with_db(root)
    parent: int | None = None
    for i in range(5):
        parent = _insert_node(conn, f"done chain goal {i}", "done", kind="goal", parent_id=parent)
    live_ids: list[int] = []
    live_parent: int | None = None
    for i in range(4):
        live_parent = _insert_node(
            conn, f"live chain goal {i}", "in_progress", kind="goal", parent_id=live_parent
        )
        live_ids.append(live_parent)
    conn.commit()
    conn.close()
    return tuple(live_ids)


class TestDryRunStructuralParity:
    def test_dry_run_and_real_run_structural_reports_agree(self, tmp_path: Path) -> None:
        dry_root = tmp_path / "dry"
        dry_root.mkdir()
        real_root = tmp_path / "real"
        real_root.mkdir()
        live_chain = _seed_chain_project(dry_root)
        _seed_chain_project(real_root)

        dry = rebalance(dry_root, dry_run=True, git=FakeGit())
        real = rebalance(real_root, git=FakeGit())

        # The DONE chain is archived by the real sweep, so neither run reports
        # it; the LIVE chain survives and both runs report exactly it.
        assert dry.chains == real.chains == (live_chain,)
        assert dry.lopsided == real.lopsided == ()


# ── #15: dry-run reap plans the dirty probe the real run performs ─────────────


class TestDryRunReapProbeParity:
    def _dirty_worktree_project(self, root: Path) -> str:
        _git(root, "init", "-b", "main")
        _git(root, "commit", "--allow-empty", "-m", "init")
        wt = root / ".worktrees" / "wt-dirty"
        _git(root, "worktree", "add", "-b", "milknado/dirty", str(wt))
        (wt / "dirty.txt").write_text("uncommitted")
        conn = _project_with_db(root)
        node = _insert_node(
            conn,
            "archived dirty",
            "done",
            archived_at=NOW,
            worktree_path=str(wt),
            branch_name="milknado/dirty",
        )
        _register_run(conn, node)
        conn.commit()
        conn.close()
        return str(wt)

    def test_dry_run_plans_preservation_for_a_dirty_worktree(self, tmp_path: Path) -> None:
        dry_root = tmp_path / "dry"
        dry_root.mkdir()
        real_root = tmp_path / "real"
        real_root.mkdir()
        self._dirty_worktree_project(dry_root)
        real_wt = self._dirty_worktree_project(real_root)

        dry = rebalance(dry_root, dry_run=True, git=GitAdapter(dry_root))
        real = rebalance(real_root, git=GitAdapter(real_root))

        # Plan matches apply: the dirty worktree is preserved in both, removed
        # in neither, and no branch deletion is promised or taken.
        assert dry.reaped == real.reaped == ()
        assert dry.branches_deleted == real.branches_deleted == ()
        assert len(dry.preserved) == 1
        assert real.preserved == (real_wt,)


# ── #11/#12: transaction rollback on restructure failure ──────────────────────


class TestTransactionRollback:
    def test_restructure_failure_leaves_no_partial_archive_stamps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _project_with_db(tmp_path)
        _insert_node(conn, "done goal", "done", kind="goal")
        conn.commit()
        conn.close()

        def boom(_conn: sqlite3.Connection) -> int:
            raise RuntimeError("simulated restructure failure")

        monkeypatch.setattr(app_rebalance, "group_orphans", boom)
        with pytest.raises(RuntimeError, match="simulated restructure failure"):
            rebalance(tmp_path, git=FakeGit())

        check = sqlite3.connect(str(tmp_path / ".milknado" / "milknado.db"))
        check.row_factory = sqlite3.Row
        assert _archived_ids(check) == set()
        check.close()


# ── GitAdapter.delete_branch against a real repository ────────────────────────


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


class TestDeleteBranchAdapter:
    def _repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "commit", "--allow-empty", "-m", "init")
        return repo

    def test_deletes_merged_branch(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        _git(repo, "branch", "merged-branch")
        GitAdapter(repo).delete_branch("merged-branch")
        probe = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", "refs/heads/merged-branch"],
            cwd=repo,
            capture_output=True,
        )
        assert probe.returncode == 1  # branch gone

    def test_unmerged_branch_refusal_surfaces_nonzero_exit(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        _git(repo, "checkout", "-b", "unmerged-branch")
        _git(repo, "commit", "--allow-empty", "-m", "unmerged work")
        _git(repo, "checkout", "main")
        adapter = GitAdapter(repo)
        with pytest.raises(GitOperationError):
            adapter.delete_branch("unmerged-branch")
        assert adapter.branch_exists("unmerged-branch")  # refusal never forces
