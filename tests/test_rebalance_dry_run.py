"""Focused rebalance regression tests."""

from __future__ import annotations

# These tests intentionally exercise private database helpers.
import sqlite3
from pathlib import Path
from typing import cast

from milknado.domains.graph.rebalance import (
    INBOX_DESCRIPTION,
    RebalanceReport,
    render_report,
)
from tests.rebalance_helpers import (
    NOW,
    FakeGit,
    archived_ids,
    insert_node,
    project_with_db,
    register_run,
    run_rebalance,
)

# ── #15: dry-run ─────────────────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_mutates_nothing_and_prefixes_report(self, tmp_path: Path) -> None:
        conn = project_with_db(tmp_path)
        done_root = insert_node(conn, "done goal", "done", kind="goal")
        orphan = insert_node(conn, "orphan", "pending")
        wt = str(tmp_path / ".worktrees" / "wt-3")
        reaped_node = insert_node(
            conn, "done wt", "done", worktree_path=wt, branch_name="milknado/3-z"
        )
        register_run(conn, reaped_node)
        conn.commit()
        conn.close()

        git = FakeGit()
        report = run_rebalance(tmp_path, dry_run=True, git=git)

        # Counts reflect the plan (both all-DONE roots are swept; the reaped
        # node's worktree is planned via the would-be-archived gate)...
        assert report.archived_roots == (done_root, reaped_node)
        assert report.moved_tasks == 1
        assert report.reaped == (wt,)
        # Exact branch targets are part of the plan, but no Git mutation runs.
        assert report.branches_deleted == ("milknado/3-z",)
        # ...and nothing happened: no git calls, no stamps, no moves, no Inbox.
        assert git.removed == [] and git.deleted_branches == [] and git.pruned == 0

        check = sqlite3.connect(str(tmp_path / ".milknado" / "milknado.db"))
        check.row_factory = sqlite3.Row
        assert archived_ids(check) == set()
        row = cast(
            sqlite3.Row,
            check.execute("SELECT parent_id FROM nodes WHERE id = ?", (orphan,)).fetchone(),
        )
        assert row["parent_id"] is None
        inbox = cast(
            sqlite3.Row | None,
            check.execute(
                "SELECT 1 FROM nodes WHERE description = ?", (INBOX_DESCRIPTION,)
            ).fetchone(),
        )
        assert inbox is None
        check.close()

        rendered = render_report(report, dry_run=True)
        lines = rendered.splitlines()
        assert len(lines) == 7
        assert all(line.startswith("[dry-run] Would ") for line in lines)

    def test_dry_run_never_opens_a_write_transaction(self, tmp_path: Path) -> None:
        # A concurrent writer holds the RESERVED lock; a read-only dry-run must
        # still compute the full plan (the old BEGIN IMMEDIATE path would fail
        # with "database is locked" here).
        conn = project_with_db(tmp_path)
        done_root = insert_node(conn, "done goal", "done", kind="goal")
        _ = insert_node(conn, "orphan", "pending")
        conn.commit()
        conn.close()

        holder = sqlite3.connect(str(tmp_path / ".milknado" / "milknado.db"))
        _ = holder.execute("BEGIN IMMEDIATE")
        _ = holder.execute(
            "INSERT INTO nodes (description, status, kind, created_at) VALUES (?, ?, ?, ?)",
            ("held", "pending", "task", NOW),
        )
        try:
            report = run_rebalance(tmp_path, dry_run=True, git=FakeGit())
        finally:
            holder.rollback()
            holder.close()
        assert report.archived_roots == (done_root,)
        assert report.moved_tasks == 1

    def test_dry_run_on_missing_project_creates_no_database(self, tmp_path: Path) -> None:
        report = run_rebalance(tmp_path, dry_run=True, git=FakeGit())
        assert report == RebalanceReport()
        assert not (tmp_path / ".milknado" / "milknado.db").exists()

    def test_dry_run_migrates_only_its_snapshot(self, tmp_path: Path) -> None:
        conn = project_with_db(tmp_path)
        _ = conn.execute("ALTER TABLE nodes DROP COLUMN archived_at")
        _ = conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()

        assert run_rebalance(tmp_path, dry_run=True, git=FakeGit()) == RebalanceReport()

        check = sqlite3.connect(str(tmp_path / ".milknado" / "milknado.db"))
        assert check.execute("PRAGMA user_version").fetchone()[0] == 2
        columns = {
            cast(str, row[1])
            for row in cast(
                list[tuple[object, ...]],
                check.execute("PRAGMA table_info(nodes)").fetchall(),
            )
        }
        assert "archived_at" not in columns
        check.close()

    def test_dry_run_plan_renders_new_inbox_when_one_would_be_created(
        self, tmp_path: Path
    ) -> None:
        conn = project_with_db(tmp_path)
        _ = insert_node(conn, "orphan", "pending")
        conn.commit()
        conn.close()

        report = run_rebalance(tmp_path, dry_run=True, git=FakeGit())
        assert report.inbox_id is not None
        assert report.inbox_created is True
        assert report.moved_tasks == 1
        assert "under Inbox (new)" in render_report(report, dry_run=True)


# ── #15: dry-run counters match the real run ──────────────────────────────────


def _seed_rebalance_project(root: Path) -> dict[str, int | str]:
    conn = project_with_db(root)
    done_root = insert_node(conn, "done goal", "done", kind="goal")
    _ = insert_node(conn, "orphan", "pending")
    wt = str(root / ".worktrees" / "wt-x")
    reaped = insert_node(conn, "done wt", "done", worktree_path=wt, branch_name="milknado/x")
    register_run(conn, reaped)
    conn.commit()
    conn.close()
    return {"done_root": done_root, "wt": wt}


class TestDryRunParity:
    def test_dry_run_counters_equal_real_run_counters(self, tmp_path: Path) -> None:
        dry_root = tmp_path / "dry"
        dry_root.mkdir()
        real_root = tmp_path / "real"
        real_root.mkdir()
        _ = _seed_rebalance_project(dry_root)
        _ = _seed_rebalance_project(real_root)

        dry = run_rebalance(dry_root, dry_run=True, git=FakeGit())
        real = run_rebalance(real_root, git=FakeGit())

        def counts(r: RebalanceReport) -> tuple[int, int, int, int]:
            return (
                len(r.archived_roots),
                r.moved_tasks,
                len(r.reaped),
                len(r.preserved),
            )

        assert counts(dry) == counts(real) == (2, 1, 1, 0)
        assert dry.branches_deleted == real.branches_deleted == ("milknado/x",)
        assert real.inbox_id is not None


# ── #12/#5: an archived Inbox never attracts live orphans (end to end) ───────


class TestArchivedInboxEndToEnd:
    def test_orphan_attaches_to_fresh_inbox_after_sweep_shelves_the_old_one(
        self, tmp_path: Path
    ) -> None:
        # The reproduced defect: Inbox fills with DONE work -> sweep archives
        # the Inbox subtree -> a new orphan must attach to a FRESH Inbox, not
        # the archived one.
        conn = project_with_db(tmp_path)
        inbox = insert_node(conn, INBOX_DESCRIPTION, "done", kind="goal")
        _ = insert_node(conn, "finished inbox task", "done", parent_id=inbox)
        orphan = insert_node(conn, "fresh live orphan", "pending")
        conn.commit()
        conn.close()

        report = run_rebalance(tmp_path, git=FakeGit())
        assert inbox in report.archived_roots  # sweep shelved the old Inbox
        assert report.moved_tasks == 1
        assert report.inbox_id is not None and report.inbox_id != inbox

        check = sqlite3.connect(str(tmp_path / ".milknado" / "milknado.db"))
        check.row_factory = sqlite3.Row
        row = cast(
            sqlite3.Row,
            check.execute("SELECT parent_id FROM nodes WHERE id = ?", (orphan,)).fetchone(),
        )
        assert row["parent_id"] == report.inbox_id
        parent = cast(
            sqlite3.Row,
            check.execute(
                "SELECT archived_at, description FROM nodes WHERE id = ?", (report.inbox_id,)
            ).fetchone(),
        )
        assert parent["archived_at"] is None
        assert parent["description"] == INBOX_DESCRIPTION
        # The shelved Inbox was never mutated.
        old = cast(
            sqlite3.Row,
            check.execute("SELECT archived_at FROM nodes WHERE id = ?", (inbox,)).fetchone(),
        )
        assert old["archived_at"] is not None
        check.close()


# ── #12/#5: dry-run predicts the fresh Inbox when the old one is swept ───────


def _seed_swept_inbox_project(root: Path) -> int:
    """An all-DONE Inbox subtree (sweep-eligible) plus one live orphan."""
    conn = project_with_db(root)
    inbox = insert_node(conn, INBOX_DESCRIPTION, "done", kind="goal")
    _ = insert_node(conn, "finished inbox task", "done", parent_id=inbox)
    _ = insert_node(conn, "fresh live orphan", "pending")
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

        dry = run_rebalance(dry_root, dry_run=True, git=FakeGit())
        real = run_rebalance(real_root, git=FakeGit())

        # The dry plan records the snapshot id but renders it as newly created.
        assert dry.inbox_id is not None
        assert dry.inbox_created is True
        plan = render_report(dry, dry_run=True)
        assert "under Inbox (new)" in plan
        assert f"under Inbox (id {dry_inbox})" not in plan

        # Parity: the real apply archived the old Inbox, moved the same orphan
        # under a FRESH Inbox, exactly as the dry plan predicted.
        assert dry.archived_roots == real.archived_roots == (real_inbox,)
        assert dry.moved_tasks == real.moved_tasks == 1
        assert real.inbox_id is not None and real.inbox_id != real_inbox
