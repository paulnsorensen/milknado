from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from milknado.adapters.git import (
    GitAdapter,
    _try_mergiraf_resolve,  # pyright: ignore[reportPrivateUsage]
)
from milknado.domains.common.errors import (
    GitOperationError,
    RebaseAbortError,
    UnlandedWorkError,
)


@pytest.fixture()
def adapter(tmp_path: Path) -> GitAdapter:
    return GitAdapter(tmp_path)


def _ok(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout, stderr)


def _fail(rc: int = 1, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], rc, stdout, stderr)


class TestCreateWorktree:
    @patch("milknado.adapters.git.subprocess.run")
    def test_calls_git_worktree_add(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok()
        wt = Path("/tmp/wt")
        result = adapter.create_worktree(wt, "feat-branch")
        mock_run.assert_called_once_with(
            ["git", "worktree", "add", "-b", "feat-branch", str(wt)],
            cwd=adapter._root,  # pyright: ignore[reportPrivateUsage]
            capture_output=True,
            text=True,
            check=True,
        )
        assert result == wt

    @patch("milknado.adapters.git.subprocess.run")
    def test_translates_transport_error(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "git", stderr="worktree failed")
        with pytest.raises(
            GitOperationError, match=r"git worktree add -b branch /tmp/wt failed"
        ) as exc_info:
            _ = adapter.create_worktree(Path("/tmp/wt"), "branch")
        assert exc_info.value.operation == "worktree add -b branch /tmp/wt"
        assert exc_info.value.detail == "worktree failed"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A real git repo on branch `feature` with one seed commit."""
    root = tmp_path / "repo"
    root.mkdir()
    _ = _git(root, "init", "-q", "-b", "feature", str(root))
    _ = _git(root, "config", "user.email", "test@test.com")
    _ = _git(root, "config", "user.name", "Test")
    _ = (root / "README.md").write_text("seed\n")
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "-qm", "seed")
    return root


def test_diff_for_review_includes_committed_and_untracked_changes(repo: Path) -> None:
    _ = (repo / "committed.py").write_text("committed = True\n")
    _ = _git(repo, "add", "committed.py")
    _ = _git(repo, "commit", "-qm", "committed change")
    base_oid = _git(repo, "rev-parse", "HEAD^").strip()
    _ = (repo / "README.md").write_text("unstaged change\n")
    _ = (repo / "untracked.py").write_text("untracked = True\n")

    diff = GitAdapter(repo).diff_for_review(repo, base_oid)

    assert "committed.py" in diff
    assert "committed = True" in diff
    assert "README.md" in diff
    assert "unstaged change" in diff
    assert "untracked.py" in diff
    assert "untracked = True" in diff


def test_diff_for_review_rejects_unexpected_untracked_diff_status(
    adapter: GitAdapter,
) -> None:
    adapter._run = MagicMock(side_effect=[_ok(), _ok("new.txt\n")])  # pyright: ignore[reportPrivateUsage]
    with patch(
        "milknado.adapters.git.subprocess.run",
        return_value=_fail(2, stderr="diff failed"),
    ):
        with pytest.raises(GitOperationError, match="diff --no-index failed"):
            _ = adapter.diff_for_review(adapter._root, "base")  # pyright: ignore[reportPrivateUsage]


def test_diff_for_review_translates_os_error(adapter: GitAdapter) -> None:
    adapter._run = MagicMock(side_effect=OSError("git unavailable"))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(GitOperationError, match="diff for review failed"):
        _ = adapter.diff_for_review(adapter._root, "base")  # pyright: ignore[reportPrivateUsage]


def _worktree_with_commit(repo: Path, name: str) -> Path:
    """A worktree on its own branch with one committed (unlanded) change."""
    wt = repo.parent / name
    _ = _git(repo, "worktree", "add", "-q", "-b", f"wb-{name}", str(wt))
    _ = (wt / f"{name}.py").write_text("x = 1\n")
    _ = _git(wt, "add", ".")
    _ = _git(wt, "commit", "-qm", f"work in {name}")
    return wt


class TestRemoveWorktreeFailClosed:
    """Fail-closed teardown against real git repos — the dirty guard, the
    two-layer landed check, and one test per merge shape (acceptance 2)."""

    def test_removes_clean_worktree_with_no_new_work(self, repo: Path) -> None:
        wt = repo.parent / "wt-clean"
        _ = _git(repo, "worktree", "add", "-q", "-b", "wb-clean", str(wt))
        GitAdapter(repo).remove_worktree(wt, "feature")
        assert not wt.exists()

    def test_refuses_dirty_worktree_naming_files(self, repo: Path) -> None:
        wt = repo.parent / "wt-dirty"
        _ = _git(repo, "worktree", "add", "-q", "-b", "wb-dirty", str(wt))
        _ = (wt / "uncommitted.py").write_text("at risk\n")
        with pytest.raises(UnlandedWorkError) as exc_info:
            GitAdapter(repo).remove_worktree(wt, "feature")
        assert str(wt) in str(exc_info.value)
        assert "uncommitted.py" in str(exc_info.value)
        assert wt.exists(), "refusal must preserve the worktree"

    def test_refuses_unlanded_commits_naming_range(self, repo: Path) -> None:
        wt = _worktree_with_commit(repo, "wt-unlanded")
        with pytest.raises(UnlandedWorkError) as exc_info:
            GitAdapter(repo).remove_worktree(wt, "feature")
        assert str(wt) in str(exc_info.value)
        assert "work in wt-unlanded" in str(exc_info.value), "diagnostic must name the commits"
        assert wt.exists()

    def test_milknado_land_path_squash_rebase_ff_removes(self, repo: Path) -> None:
        """Milknado's own land shape: squash_and_commit + rebase + ff-only makes
        the worktree HEAD literally equal the feature tip — no false refusal."""
        adapter = GitAdapter(repo)
        wt = _worktree_with_commit(repo, "wt-land")
        assert adapter.squash_and_commit(wt, "feature", "feat: squashed") is True
        assert adapter.rebase(wt, "feature").success is True
        adapter.fast_forward("wb-wt-land")
        adapter.remove_worktree(wt, "feature")
        assert not wt.exists()

    def test_conflicting_target_refuses_not_guesses(self, repo: Path) -> None:
        """When the target diverged incompatibly, the worktree HEAD is not an
        ancestor — refuse, never destroy on a guess."""
        wt = _worktree_with_commit(repo, "wt-conflict")
        # Diverge the feature branch with conflicting content in the same file.
        _ = (repo / "wt-conflict.py").write_text("x = 2  # conflicts\n")
        _ = _git(repo, "add", ".")
        _ = _git(repo, "commit", "-qm", "feature diverges")
        with pytest.raises(UnlandedWorkError):
            GitAdapter(repo).remove_worktree(wt, "feature")
        assert wt.exists()

    def test_stale_non_worktree_dir_raises_non_refusal(self, repo: Path) -> None:
        """A path that exists but is NOT a registered worktree (stale dir after
        a prune / crashed create) resolves git commands against the PARENT
        repo — without the toplevel guard, a dirty project root would produce a
        FALSE UnlandedWorkError naming the root's files. It must raise a plain
        ValueError (non-refusal: managers warn-and-swallow it, as the old
        CalledProcessError path did) and never touch the stale dir."""
        _ = (repo / "root-dirt.py").write_text("uncommitted root work\n")
        stale = repo / "milknado-stale"
        stale.mkdir()
        with pytest.raises(ValueError, match="not a registered worktree"):
            GitAdapter(repo).remove_worktree(stale, "feature")
        assert stale.exists()

    def test_ignored_scaffolding_does_not_block_removal(self, repo: Path) -> None:
        """RALPH.md / .ralph-logs are git-ignored in real worktrees — they must
        not count as dirt (verified against git 2.53: plain `worktree remove`
        deletes ignored files without --force)."""
        _ = (repo / ".gitignore").write_text("RALPH.md\n.ralph-logs/\n")
        _ = _git(repo, "add", ".gitignore")
        _ = _git(repo, "commit", "-qm", "ignore scaffolding")
        wt = repo.parent / "wt-scaffold"
        _ = _git(repo, "worktree", "add", "-q", "-b", "wb-scaffold", str(wt))
        _ = (wt / "RALPH.md").write_text("# scaffolding\n")
        (wt / ".ralph-logs").mkdir()
        _ = (wt / ".ralph-logs" / "run.log").write_text("log\n")
        GitAdapter(repo).remove_worktree(wt, "feature")
        assert not wt.exists()


class TestForceRemoveWorktree:
    def test_destroys_dirty_and_unlanded_work(self, repo: Path) -> None:
        """The explicitly-named destructive variant: discards what the
        fail-closed path refuses."""
        wt = _worktree_with_commit(repo, "wt-doomed")
        _ = (wt / "dirty.py").write_text("also at risk\n")
        GitAdapter(repo).force_remove_worktree(wt)
        assert not wt.exists()

    @patch("milknado.adapters.git.subprocess.run")
    def test_passes_force_flag(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok()
        _ = adapter.force_remove_worktree(Path("/tmp/wt"))
        mock_run.assert_called_once_with(
            ["git", "worktree", "remove", "--force", "/tmp/wt"],
            cwd=adapter._root,  # pyright: ignore[reportPrivateUsage]
            capture_output=True,
            text=True,
            check=True,
        )


class TestPruneWorktrees:
    @patch("milknado.adapters.git.subprocess.run")
    def test_calls_git_worktree_prune(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok()
        adapter.prune_worktrees()
        mock_run.assert_called_once_with(
            ["git", "worktree", "prune"],
            cwd=adapter._root,  # pyright: ignore[reportPrivateUsage]
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("milknado.adapters.git.subprocess.run")
    def test_translates_transport_error(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.side_effect = OSError("git unavailable")
        with pytest.raises(GitOperationError, match="git worktree prune failed") as exc_info:
            adapter.prune_worktrees()
        assert exc_info.value.operation == "worktree prune"
        assert exc_info.value.detail == "git unavailable"


class TestRebase:
    @patch("milknado.adapters.git.subprocess.run")
    def test_successful_rebase(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok()
        result = adapter.rebase(Path("/tmp/wt"), "main")
        assert result.success is True

    @patch("milknado.adapters.git._try_mergiraf_resolve", return_value=False)
    @patch("milknado.adapters.git.subprocess.run")
    def test_failed_rebase_no_mergiraf_aborts_and_returns_false(
        self, mock_run: MagicMock, _mergiraf: MagicMock, adapter: GitAdapter
    ) -> None:
        conflict_out = "CONFLICT (content): Merge conflict in src/foo.py\n"
        mock_run.side_effect = [_fail(1, conflict_out, ""), _ok()]

        result = adapter.rebase(Path("/tmp/wt"), "main")

        assert result.success is False
        assert mock_run.call_count == 2
        abort_call = mock_run.call_args_list[1]
        assert "--abort" in abort_call.args[0]

    @patch("milknado.adapters.git._try_mergiraf_resolve", return_value=False)
    @patch("milknado.adapters.git.subprocess.run")
    def test_rebase_abort_failure_raises(
        self, mock_run: MagicMock, _mergiraf: MagicMock, adapter: GitAdapter
    ) -> None:
        conflict_out = "CONFLICT (content): Merge conflict in foo.py\n"
        mock_run.side_effect = [
            _fail(1, conflict_out, ""),
            _fail(1, "", "fatal: no rebase in progress"),
        ]

        with pytest.raises(RebaseAbortError) as exc_info:
            _ = adapter.rebase(Path("/tmp/wt"), "main")

        assert "fatal: no rebase in progress" in exc_info.value.stderr

    @patch("milknado.adapters.git._try_mergiraf_resolve", return_value=True)
    @patch("milknado.adapters.git.subprocess.run")
    def test_mergiraf_happy_path_returns_success(
        self, mock_run: MagicMock, _mergiraf: MagicMock, adapter: GitAdapter
    ) -> None:
        conflict_out = "CONFLICT (content): Merge conflict in src/foo.py\n"
        # rebase fails, then git add -A, then rebase --continue both succeed
        mock_run.side_effect = [
            _fail(1, conflict_out, ""),  # initial rebase
            _ok(),  # git add -A
            _ok(),  # git rebase --continue
        ]

        result = adapter.rebase(Path("/tmp/wt"), "main")
        assert result.success is True

    @patch("milknado.adapters.git._try_mergiraf_resolve", return_value=True)
    @patch("milknado.adapters.git.subprocess.run")
    def test_mergiraf_continue_fails_falls_through_to_abort(
        self, mock_run: MagicMock, _mergiraf: MagicMock, adapter: GitAdapter
    ) -> None:
        conflict_out = "CONFLICT (content): Merge conflict in src/foo.py\n"
        mock_run.side_effect = [
            _fail(1, conflict_out, ""),  # initial rebase
            _ok(),  # git add -A
            _fail(1, "", "conflict"),  # rebase --continue fails
            _ok(),  # rebase --abort succeeds
        ]

        result = adapter.rebase(Path("/tmp/wt"), "main")
        assert result.success is False
        abort_call = mock_run.call_args_list[3]
        assert "--abort" in abort_call.args[0]

    @patch("milknado.adapters.git.subprocess.run")
    def test_conflict_files_captured_in_result(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        conflict_out = (
            "CONFLICT (content): Merge conflict in src/a.py\n"
            "CONFLICT (content): Merge conflict in src/b.py\n"
        )
        with patch("milknado.adapters.git._try_mergiraf_resolve", return_value=False):
            mock_run.side_effect = [_fail(1, conflict_out, ""), _ok()]
            result = adapter.rebase(Path("/tmp/wt"), "main")

        assert result.success is False
        assert "src/a.py" in result.conflicting_files
        assert "src/b.py" in result.conflicting_files


class TestTryMegirafResolve:
    @patch("milknado.adapters.git.shutil.which", return_value=None)
    def test_returns_false_when_binary_missing(self, _which: MagicMock) -> None:
        assert _try_mergiraf_resolve(Path("/wt"), ("foo.py",)) is False

    @patch("milknado.adapters.git.subprocess.run")
    @patch("milknado.adapters.git.shutil.which", return_value="/usr/bin/mergiraf")
    def test_returns_true_when_all_files_resolve(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _ok()
        result = _try_mergiraf_resolve(Path("/wt"), ("a.py", "b.py"))
        assert result is True
        assert mock_run.call_count == 2

    @patch("milknado.adapters.git.subprocess.run")
    @patch("milknado.adapters.git.shutil.which", return_value="/usr/bin/mergiraf")
    def test_returns_false_when_any_file_fails(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = [_ok(), _fail(1)]
        result = _try_mergiraf_resolve(Path("/wt"), ("a.py", "b.py"))
        assert result is False

    @patch("milknado.adapters.git.shutil.which", return_value="/usr/bin/mergiraf")
    def test_empty_files_returns_true(self, _which: MagicMock) -> None:
        # No files to resolve — no subprocess calls needed
        with patch("milknado.adapters.git.subprocess.run") as mock_run:
            result = _try_mergiraf_resolve(Path("/wt"), ())
            assert result is True
            mock_run.assert_not_called()


class TestCurrentBranch:
    @patch("milknado.adapters.git.subprocess.run")
    def test_returns_branch_name(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok("feat/cool\n")
        assert adapter.current_branch() == "feat/cool"


class TestSquashAndCommit:
    @patch("milknado.adapters.git.subprocess.run")
    def test_squashes_and_commits_when_staged(
        self, mock_run: MagicMock, adapter: GitAdapter, tmp_path: Path
    ) -> None:
        # Calls: git add -A, merge-base, reset --soft, diff --cached (1=staged), commit
        mock_run.side_effect = [
            _ok(),  # git add -A
            _ok("abc123\n"),  # merge-base
            _ok(),  # reset --soft
            _fail(1),  # diff --cached --quiet returns 1 means staged changes exist
            _ok(),  # commit
        ]
        _ = adapter.squash_and_commit(tmp_path, "main", "feat: squashed")
        commit_call = mock_run.call_args_list[-1]
        assert "commit" in commit_call.args[0]
        assert "feat: squashed" in commit_call.args[0]

    @patch("milknado.adapters.git.subprocess.run")
    def test_skips_commit_when_nothing_staged(
        self, mock_run: MagicMock, adapter: GitAdapter, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            _ok(),  # git add -A
            _ok("abc123\n"),  # merge-base
            _ok(),  # reset --soft
            _ok(),  # diff --cached --quiet returns 0 = nothing staged
        ]
        _ = adapter.squash_and_commit(tmp_path, "main", "feat: squashed")
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert not any("commit" in c for c in calls)  # pyright: ignore[reportAny]

    @patch("milknado.adapters.git.subprocess.run")
    def test_merge_base_failure_skips_reset(
        self, mock_run: MagicMock, adapter: GitAdapter, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            _ok(),  # git add -A
            subprocess.CompletedProcess([], 1, "", "not a git repo"),  # merge-base raises
            _ok(),  # diff --cached --quiet returns 0 = nothing staged
        ]
        # merge-base failure swallows CalledProcessError, falls through to diff check
        # Since we need check=True to raise, let's use side_effect properly
        mock_run.side_effect = [
            _ok(),  # git add -A (called via _run with check=True)
            subprocess.CalledProcessError(1, "git"),  # merge-base raises CalledProcessError
            _ok(),  # diff --cached --quiet returns 0
        ]
        _ = adapter.squash_and_commit(tmp_path, "main", "msg")
        # Should not raise; commit skipped because nothing staged
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert not any("commit" in c for c in calls if isinstance(c, list))  # pyright: ignore[reportAny]


class TestBranchExists:
    """Real-git existence check used by dispatch relocation to keep path and
    branch slots in lockstep."""

    def test_true_for_live_branch_false_for_absent(self, repo: Path) -> None:
        adapter = GitAdapter(repo)
        assert adapter.branch_exists("feature") is True
        assert adapter.branch_exists("no-such-branch") is False

    def test_recognizes_namespaced_relocation_branch(self, repo: Path) -> None:
        _ = _git(repo, "branch", "milknado/1-slug-2")
        adapter = GitAdapter(repo)
        assert adapter.branch_exists("milknado/1-slug-2") is True


class TestAtomicRefUpdate:
    def test_resolve_ref_returns_exact_oid(self, repo: Path) -> None:
        git = GitAdapter(repo)
        assert git.resolve_ref("feature") == _git(repo, "rev-parse", "feature").strip()

    def test_checked_out_ref_update_synchronizes_worktree(self, repo: Path) -> None:
        old_oid = _git(repo, "rev-parse", "HEAD").strip()
        _ = (repo / "landed.txt").write_text("landed")
        _ = _git(repo, "add", "landed.txt")
        _ = _git(repo, "commit", "-qm", "next")
        new_oid = _git(repo, "rev-parse", "HEAD").strip()
        _ = _git(repo, "reset", "--hard", old_oid)

        GitAdapter(repo).compare_and_swap_ref("refs/heads/feature", old_oid, new_oid)

        assert _git(repo, "rev-parse", "HEAD").strip() == new_oid
        assert (repo / "landed.txt").read_text() == "landed"
        assert _git(repo, "status", "--porcelain") == ""

    def test_dirty_checked_out_ref_fails_before_update(self, repo: Path) -> None:
        old_oid = _git(repo, "rev-parse", "feature").strip()
        new_oid = _git(
            repo, "commit-tree", f"{old_oid}^{{tree}}", "-p", old_oid, "-m", "next"
        ).strip()
        _ = (repo / "README.md").write_text("dirty")

        with pytest.raises(GitOperationError, match="tracked changes"):
            GitAdapter(repo).compare_and_swap_ref("feature", old_oid, new_oid)

        assert _git(repo, "rev-parse", "feature").strip() == old_oid

    @patch("milknado.adapters.git.subprocess.run")
    def test_symbolic_ref_transport_failure_is_domain_error(
        self, run: MagicMock, repo: Path
    ) -> None:
        run.return_value = subprocess.CompletedProcess(
            [], 128, stdout="", stderr="symbolic failure"
        )
        with pytest.raises(GitOperationError, match="symbolic failure"):
            GitAdapter(repo).compare_and_swap_ref("feature", "old", "new")

    @patch("milknado.adapters.git.subprocess.run")
    def test_diff_index_transport_failure_is_domain_error(
        self, run: MagicMock, repo: Path
    ) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="refs/heads/feature\n", stderr=""),
            subprocess.CompletedProcess([], 128),
        ]
        with pytest.raises(GitOperationError, match="git exited 128"):
            GitAdapter(repo).compare_and_swap_ref("feature", "old", "new")

    def test_checked_out_head_must_still_match_expected(self, repo: Path) -> None:
        actual = _git(repo, "rev-parse", "HEAD").strip()
        stale = _git(repo, "commit-tree", f"{actual}^{{tree}}", "-m", "stale").strip()
        with pytest.raises(GitOperationError, match="no longer matches expected OID"):
            GitAdapter(repo).compare_and_swap_ref("feature", stale, actual)

    def test_switched_branch_ref_update_does_not_touch_worktree(self, repo: Path) -> None:
        old_oid = _git(repo, "rev-parse", "feature").strip()
        _ = _git(repo, "branch", "other")
        _ = _git(repo, "switch", "other")
        new_oid = _git(
            repo, "commit-tree", f"{old_oid}^{{tree}}", "-p", old_oid, "-m", "next"
        ).strip()

        GitAdapter(repo).compare_and_swap_ref("feature", old_oid, new_oid)

        assert _git(repo, "rev-parse", "HEAD").strip() == old_oid
        assert _git(repo, "rev-parse", "feature").strip() == new_oid

    @patch("milknado.adapters.git.subprocess.run")
    def test_sync_failure_rolls_back_tree_and_ref(self, run: MagicMock, repo: Path) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="refs/heads/feature\n", stderr=""),
            subprocess.CompletedProcess([], 0),
        ]
        adapter = GitAdapter(repo)
        adapter._run = MagicMock(  # pyright: ignore[reportPrivateUsage]
            side_effect=[
                _ok("old\n"),
                _ok(),
                GitOperationError("read-tree", "collision"),
                _ok(),
                _ok(),
            ]
        )

        with pytest.raises(GitOperationError, match="collision"):
            adapter.compare_and_swap_ref("feature", "old", "new")

        assert adapter._run.call_args_list[-2:] == [  # pyright: ignore[reportPrivateUsage]
            call(["read-tree", "-u", "-m", "new", "old"]),
            call(["update-ref", "refs/heads/feature", "old", "new"]),
        ]

    @patch("milknado.adapters.git.subprocess.run")
    def test_rollback_failure_is_logged(
        self, run: MagicMock, repo: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="refs/heads/feature\n", stderr=""),
            subprocess.CompletedProcess([], 0),
        ]
        adapter = GitAdapter(repo)
        adapter._run = MagicMock(  # pyright: ignore[reportPrivateUsage]
            side_effect=[
                _ok("old\n"),
                _ok(),
                GitOperationError("read-tree", "collision"),
                GitOperationError("read-tree", "rollback collision"),
            ]
        )

        with caplog.at_level("ERROR"), pytest.raises(GitOperationError, match="collision"):
            adapter.compare_and_swap_ref("feature", "old", "new")

        assert "failed to roll back ref and worktree" in caplog.text
