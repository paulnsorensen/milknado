from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.adapters.git import GitAdapter, _try_mergiraf_resolve
from milknado.domains.common.errors import RebaseAbortError


@pytest.fixture()
def adapter(tmp_path: Path) -> GitAdapter:
    return GitAdapter(tmp_path)


def _ok(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout, stderr)


def _fail(rc: int = 1, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], rc, stdout, stderr)


class TestCreateWorktree:
    @patch("milknado.adapters.git.subprocess.run")
    def test_cleans_stale_state_then_adds_with_force_create(
        self, mock_run: MagicMock, adapter: GitAdapter, tmp_path: Path
    ) -> None:
        mock_run.return_value = _ok()
        wt = tmp_path / "wt"  # does not exist -> rmtree branch skipped
        result = adapter.create_worktree(wt, "feat-branch")
        calls = [c.args[0] for c in mock_run.call_args_list]
        # best-effort stale-worktree removal, prune, then create-or-reset branch
        assert ["git", "worktree", "remove", "--force", str(wt)] in calls
        assert ["git", "worktree", "prune"] in calls
        # -B (not -b) so a leftover branch is reset rather than fatally colliding
        assert ["git", "worktree", "add", "-B", "feat-branch", str(wt)] in calls
        assert result == wt

    @patch("milknado.adapters.git.subprocess.run")
    def test_reclaims_stale_worktree_dir_with_gitlink(
        self, mock_run: MagicMock, adapter: GitAdapter, tmp_path: Path
    ) -> None:
        # A leftover dir that IS a git worktree (has a .git gitlink) is safe to
        # rmtree-reclaim when `git worktree remove` left it behind.
        mock_run.return_value = _ok()
        wt = tmp_path / "wt"
        wt.mkdir()
        (wt / ".git").write_text("gitdir: /somewhere\n", encoding="utf-8")
        result = adapter.create_worktree(wt, "feat-branch")
        assert result == wt
        assert not wt.exists()  # rmtree reclaimed it before re-add

    def test_refuses_to_delete_ordinary_directory_collision(
        self, adapter: GitAdapter, tmp_path: Path
    ) -> None:
        # worktree_pattern is user-configurable; a collision with an ordinary
        # project directory must fail loudly, never be recursively deleted.
        wt = tmp_path / "wt"
        wt.mkdir()
        precious = wt / "important.txt"
        precious.write_text("do not delete", encoding="utf-8")
        with patch("milknado.adapters.git.subprocess.run", return_value=_ok()):
            with pytest.raises(RuntimeError, match="not a git worktree"):
                adapter.create_worktree(wt, "feat-branch")
        assert precious.exists()  # untouched

    @patch("milknado.adapters.git.subprocess.run")
    def test_propagates_error_on_add(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        # remove (best-effort, no check) and prune succeed; the add call raises
        mock_run.side_effect = [_ok(), _ok(), subprocess.CalledProcessError(1, "git")]
        with pytest.raises(subprocess.CalledProcessError):
            adapter.create_worktree(Path("/tmp/wt-x"), "branch")

    def test_recovers_when_branch_already_exists(self, tmp_path: Path) -> None:
        """Reproduces node 5's failure: a worktree is created, the dir removed but
        the branch left behind, then the node is re-dispatched. With `-b` this
        raised exit 255 and failed the node; `-B` must recreate it cleanly."""
        repo = tmp_path / "repo"
        repo.mkdir()

        def git(*args: str) -> None:
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

        git("init", "-q", "-b", "main")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "Test")
        (repo / "f.txt").write_text("x", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "init")

        adapter = GitAdapter(repo)
        wt = repo / "milknado-5-wt"

        adapter.create_worktree(wt, "milknado/5-x")
        assert wt.exists()

        # Interrupted run: worktree dir gone, branch milknado/5-x lingers.
        adapter.remove_worktree(wt)
        branches = subprocess.run(
            ["git", "branch", "--list", "milknado/5-x"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).stdout
        assert "milknado/5-x" in branches  # stale branch present

        # Re-dispatch must NOT raise and must recreate the worktree.
        adapter.create_worktree(wt, "milknado/5-x")
        assert wt.exists()


class TestRemoveWorktree:
    @patch("milknado.adapters.git.subprocess.run")
    def test_calls_git_worktree_remove(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok()
        adapter.remove_worktree(Path("/tmp/wt"))
        mock_run.assert_called_once_with(
            ["git", "worktree", "remove", "--force", "/tmp/wt"],
            cwd=adapter._root,
            capture_output=True,
            text=True,
            check=True,
        )


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
            adapter.rebase(Path("/tmp/wt"), "main")

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


class TestCommitAll:
    @patch("milknado.adapters.git.subprocess.run")
    def test_stages_and_commits(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok()
        wt = Path("/tmp/wt")
        adapter.commit_all(wt, "fix: something")
        assert mock_run.call_count == 2
        add_call = mock_run.call_args_list[0]
        assert add_call.args[0] == ["git", "add", "-A"]
        commit_call = mock_run.call_args_list[1]
        assert commit_call.args[0] == ["git", "commit", "-m", "fix: something"]


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
        adapter.squash_and_commit(tmp_path, "main", "feat: squashed")
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
        adapter.squash_and_commit(tmp_path, "main", "feat: squashed")
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert not any("commit" in c for c in calls)

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
        adapter.squash_and_commit(tmp_path, "main", "msg")
        # Should not raise; commit skipped because nothing staged
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert not any("commit" in c for c in calls if isinstance(c, list))


class TestBranchExists:
    @patch("milknado.adapters.git.subprocess.run")
    def test_returns_true_when_branch_exists(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.return_value = _ok("abc123\n")
        assert adapter.branch_exists("milknado/1-task") is True
        args = mock_run.call_args[0][0]
        assert args == ["git", "rev-parse", "--verify", "refs/heads/milknado/1-task"]

    @patch("milknado.adapters.git.subprocess.run")
    def test_returns_false_when_branch_missing(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.return_value = _fail(128)
        assert adapter.branch_exists("milknado/99-gone") is False

    def test_real_repo_existing_branch_returns_true(self, tmp_path: Path) -> None:
        import subprocess as sp

        repo = tmp_path / "repo"
        repo.mkdir()

        def git(*args: str) -> None:
            sp.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

        git("init", "-q", "-b", "main")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "Test")
        (repo / "f.txt").write_text("x", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "init")
        git("branch", "feature/foo")

        a = GitAdapter(repo)
        assert a.branch_exists("feature/foo") is True
        assert a.branch_exists("feature/nonexistent") is False


class TestIsAncestor:
    @patch("milknado.adapters.git.subprocess.run")
    def test_returns_true_when_ancestor(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok()  # exit 0 = is ancestor
        assert adapter.is_ancestor("milknado/1-task", "main") is True
        args = mock_run.call_args[0][0]
        assert args == ["git", "merge-base", "--is-ancestor", "milknado/1-task", "main"]

    @patch("milknado.adapters.git.subprocess.run")
    def test_returns_false_when_not_ancestor(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.return_value = _fail(1)  # exit 1 = not ancestor
        assert adapter.is_ancestor("milknado/1-task", "main") is False

    def test_real_repo_ancestor_detection(self, tmp_path: Path) -> None:
        import subprocess as sp

        repo = tmp_path / "repo"
        repo.mkdir()

        def git(*args: str) -> str:
            r = sp.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
            return r.stdout.strip()

        git("init", "-q", "-b", "main")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "Test")
        (repo / "a.txt").write_text("a", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "init")
        git("checkout", "-qb", "feature/x")
        (repo / "b.txt").write_text("b", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "feat")
        git("checkout", "-q", "main")
        git("merge", "--ff-only", "feature/x")

        a = GitAdapter(repo)
        # feature/x is now merged into main (fast-forward)
        assert a.is_ancestor("feature/x", "main") is True
        # A new branch with uncommitted work is NOT an ancestor of main
        git("checkout", "-qb", "feature/y")
        (repo / "c.txt").write_text("c", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "feat-y")
        git("checkout", "-q", "main")
        assert a.is_ancestor("feature/y", "main") is False
