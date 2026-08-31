"""Focused rebalance regression tests."""

from __future__ import annotations

# These tests intentionally exercise private database helpers.
from pathlib import Path

from milknado.adapters.git import GitAdapter
from milknado.domains.graph.rebalance import (
    render_report,
)
from tests.rebalance_helpers import (
    NOW,
    FakeGit,
    _git,  # pyright: ignore[reportPrivateUsage]
    _insert_node,  # pyright: ignore[reportPrivateUsage]
    _project_with_db,  # pyright: ignore[reportPrivateUsage]
    _register_run,  # pyright: ignore[reportPrivateUsage]
    run_rebalance,
)

# ── #14: reap ────────────────────────────────────────────────────────────────


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
        # No run record is still eligible; only an actively running run blocks reap.
        _ = _insert_node(
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
        report = run_rebalance(tmp_path, sweep=False, restructure=False, git=git)

        assert git.removed == [
            tmp_path / ".worktrees" / "wt-1",
            tmp_path / "elsewhere",
        ]
        assert git.deleted_branches == ["milknado/1-wt", "stray"]
        assert git.pruned == 1
        assert report.reaped == (
            str(tmp_path / ".worktrees" / "wt-1"),
            str(tmp_path / "elsewhere"),
        )
        assert report.branches_deleted == ("milknado/1-wt", "stray")
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
        report = run_rebalance(tmp_path, sweep=False, restructure=False, git=git)
        assert git.removed == [] and git.deleted_branches == [] and git.pruned == 1
        assert report.reaped == () and report.branches_deleted == () and report.preserved == ()

    def test_archived_worktree_without_run_record_is_reaped(self, tmp_path: Path) -> None:
        """Archive state is durable teardown evidence even without run history."""
        conn = _project_with_db(tmp_path)
        _ = _insert_node(
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
        report = run_rebalance(tmp_path, sweep=False, restructure=False, git=git)

        assert git.removed == [tmp_path / "elsewhere"]
        assert git.deleted_branches == ["stray"]
        assert report.reaped == (str(tmp_path / "elsewhere"),)
        assert report.branches_deleted == ("stray",)
        assert str(tmp_path / "elsewhere") in render_report(report)

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
        report = run_rebalance(tmp_path, sweep=False, restructure=False, git=git)

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

        report = run_rebalance(tmp_path, sweep=False, restructure=False, git=GitAdapter(repo))
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
        report = run_rebalance(tmp_path, sweep=False, restructure=False, git=git)
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
        report = run_rebalance(tmp_path, sweep=False, restructure=False, git=git)
        assert git.removed == [] and git.deleted_branches == [] and git.pruned == 1
        assert report.reaped == () and report.branches_deleted == () and report.preserved == ()

    def test_running_run_overrides_historical_terminal_run(self, tmp_path: Path) -> None:
        conn = _project_with_db(tmp_path)
        node = _insert_node(
            conn,
            "archived with active rerun",
            "done",
            archived_at=NOW,
            worktree_path=str(tmp_path / ".worktrees" / "wt-rerun"),
            branch_name="milknado/9-rerun",
        )
        _register_run(conn, node, status="completed")
        _ = conn.execute(
            "INSERT INTO runs (run_id, node_id, status, log_path, started_at) "
            + "VALUES (?, ?, ?, '', ?)",
            ("run-active", node, "running", NOW),
        )
        conn.commit()
        conn.close()

        git = FakeGit()
        report = run_rebalance(tmp_path, sweep=False, restructure=False, git=git)
        assert git.removed == []
        assert git.pruned == 1
        assert git.deleted_branches == []
        assert report.reaped == ()

    def test_archived_node_without_worktree_is_skipped(self, tmp_path: Path) -> None:
        conn = _project_with_db(tmp_path)
        node = _insert_node(
            conn, "archived no worktree", "done", archived_at=NOW, branch_name="milknado/9-none"
        )
        _register_run(conn, node)
        conn.commit()
        conn.close()

        git = FakeGit()
        report = run_rebalance(tmp_path, sweep=False, restructure=False, git=git)
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
        _ = run_rebalance(tmp_path, dry_run=True, sweep=False, restructure=False, git=git)
        assert git.pruned == 0
