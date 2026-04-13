from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.adapters.git import GitAdapter


@pytest.fixture()
def adapter(tmp_path: Path) -> GitAdapter:
    return GitAdapter(tmp_path)


class TestCreateWorktree:
    @patch("milknado.adapters.git.subprocess.run")
    def test_calls_git_worktree_add(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
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
    def test_propagates_error(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        with pytest.raises(subprocess.CalledProcessError):
            adapter.create_worktree(Path("/tmp/wt"), "branch")


class TestRemoveWorktree:
    @patch("milknado.adapters.git.subprocess.run")
    def test_calls_git_worktree_remove(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
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
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
        wt = Path("/tmp/wt")
        assert adapter.rebase(wt, "main") is True

    @patch("milknado.adapters.git.subprocess.run")
    def test_failed_rebase_aborts_and_returns_false(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        fail = subprocess.CompletedProcess([], 1, "", "conflict")
        success = subprocess.CompletedProcess([], 0, "", "")
        mock_run.side_effect = [fail, success]

        assert adapter.rebase(Path("/tmp/wt"), "main") is False
        assert mock_run.call_count == 2
        abort_call = mock_run.call_args_list[1]
        assert "rebase" in abort_call.args[0]
        assert "--abort" in abort_call.args[0]


class TestCurrentBranch:
    @patch("milknado.adapters.git.subprocess.run")
    def test_returns_branch_name(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            [], 0, "feat/cool\n", ""
        )
        assert adapter.current_branch() == "feat/cool"


class TestCommitAll:
    @patch("milknado.adapters.git.subprocess.run")
    def test_stages_and_commits(
        self, mock_run: MagicMock, adapter: GitAdapter
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
        wt = Path("/tmp/wt")
        adapter.commit_all(wt, "fix: something")
        assert mock_run.call_count == 2
        add_call = mock_run.call_args_list[0]
        assert add_call.args[0] == ["git", "add", "-A"]
        commit_call = mock_run.call_args_list[1]
        assert commit_call.args[0] == ["git", "commit", "-m", "fix: something"]
