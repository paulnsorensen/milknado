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
    def test_calls_git_worktree_add(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.return_value = _ok()
        wt = Path("/tmp/wt")
        result = adapter.create_worktree(wt, "feat-branch")
        mock_run.assert_called_once_with(
            ["git", "worktree", "add", "-b", "feat-branch", str(wt)],
            cwd=adapter._root,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result == wt

    @patch("milknado.adapters.git.subprocess.run")
    def test_propagates_error(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        with pytest.raises(subprocess.CalledProcessError):
            adapter.create_worktree(Path("/tmp/wt"), "branch")


class TestRemoveWorktree:
    @patch("milknado.adapters.git.subprocess.run")
    def test_calls_git_worktree_remove(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
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
    def test_successful_rebase(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
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
            _ok(),                       # git add -A
            _ok(),                       # git rebase --continue
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
            _ok(),                       # git add -A
            _fail(1, "", "conflict"),    # rebase --continue fails
            _ok(),                       # rebase --abort succeeds
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
    def test_returns_branch_name(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.return_value = _ok("feat/cool\n")
        assert adapter.current_branch() == "feat/cool"


class TestCommitAll:
    @patch("milknado.adapters.git.subprocess.run")
    def test_stages_and_commits(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
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
            _ok(),            # git add -A
            _ok("abc123\n"),  # merge-base
            _ok(),            # reset --soft
            _fail(1),         # diff --cached --quiet returns 1 means staged changes exist
            _ok(),            # commit
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
            _ok(),            # git add -A
            _ok("abc123\n"),  # merge-base
            _ok(),            # reset --soft
            _ok(),            # diff --cached --quiet returns 0 = nothing staged
        ]
        adapter.squash_and_commit(tmp_path, "main", "feat: squashed")
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert not any("commit" in c for c in calls)

    @patch("milknado.adapters.git.subprocess.run")
    def test_merge_base_failure_skips_reset(
        self, mock_run: MagicMock, adapter: GitAdapter, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            _ok(),    # git add -A
            subprocess.CompletedProcess([], 1, "", "not a git repo"),  # merge-base raises
            _ok(),    # diff --cached --quiet returns 0 = nothing staged
        ]
        # merge-base failure swallows CalledProcessError, falls through to diff check
        # Since we need check=True to raise, let's use side_effect properly
        mock_run.side_effect = [
            _ok(),    # git add -A (called via _run with check=True)
            subprocess.CalledProcessError(1, "git"),  # merge-base raises CalledProcessError
            _ok(),    # diff --cached --quiet returns 0
        ]
        adapter.squash_and_commit(tmp_path, "main", "msg")
        # Should not raise; commit skipped because nothing staged
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert not any("commit" in c for c in calls if isinstance(c, list))
