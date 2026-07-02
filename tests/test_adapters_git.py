from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.adapters.git import GitAdapter, _try_mergiraf_resolve
from milknado.domains.common.errors import RebaseAbortError, UnlandedWorkError


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
    _git(root, "init", "-q", "-b", "feature", str(root))
    _git(root, "config", "user.email", "test@test.com")
    _git(root, "config", "user.name", "Test")
    (root / "README.md").write_text("seed\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "seed")
    return root


def _worktree_with_commit(repo: Path, name: str) -> Path:
    """A worktree on its own branch with one committed (unlanded) change."""
    wt = repo.parent / name
    _git(repo, "worktree", "add", "-q", "-b", f"wb-{name}", str(wt))
    (wt / f"{name}.py").write_text("x = 1\n")
    _git(wt, "add", ".")
    _git(wt, "commit", "-qm", f"work in {name}")
    return wt


class TestRemoveWorktreeFailClosed:
    """Fail-closed teardown against real git repos — the dirty guard, the
    two-layer landed check, and one test per merge shape (acceptance 2)."""

    def test_removes_clean_worktree_with_no_new_work(self, repo: Path) -> None:
        wt = repo.parent / "wt-clean"
        _git(repo, "worktree", "add", "-q", "-b", "wb-clean", str(wt))
        GitAdapter(repo).remove_worktree(wt, "feature")
        assert not wt.exists()

    def test_refuses_dirty_worktree_naming_files(self, repo: Path) -> None:
        wt = repo.parent / "wt-dirty"
        _git(repo, "worktree", "add", "-q", "-b", "wb-dirty", str(wt))
        (wt / "uncommitted.py").write_text("at risk\n")
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

    def test_synthetic_squash_merge_recognized_as_landed(self, repo: Path) -> None:
        """External squash-merge shape: the feature branch carries the same
        content under a different SHA. Ancestor containment fails, the
        merge-tree content-equality fallback must recognize it — no false
        refusal (the git-cherry / plain-ancestor failure mode)."""
        wt = _worktree_with_commit(repo, "wt-squash")
        _git(repo, "merge", "--squash", "-q", "wb-wt-squash")
        _git(repo, "commit", "-qm", "squash-landed")
        GitAdapter(repo).remove_worktree(wt, "feature")
        assert not wt.exists()

    def test_conflicting_target_refuses_not_guesses(self, repo: Path) -> None:
        """When the merge-tree probe conflicts (target diverged incompatibly),
        the check is inconclusive — refuse, never destroy on a guess."""
        wt = _worktree_with_commit(repo, "wt-conflict")
        # Diverge the feature branch with conflicting content in the same file.
        (repo / "wt-conflict.py").write_text("x = 2  # conflicts\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "feature diverges")
        with pytest.raises(UnlandedWorkError):
            GitAdapter(repo).remove_worktree(wt, "feature")
        assert wt.exists()

    def test_ignored_scaffolding_does_not_block_removal(self, repo: Path) -> None:
        """RALPH.md / .ralph-logs are git-ignored in real worktrees — they must
        not count as dirt (verified against git 2.53: plain `worktree remove`
        deletes ignored files without --force)."""
        (repo / ".gitignore").write_text("RALPH.md\n.ralph-logs/\n")
        _git(repo, "add", ".gitignore")
        _git(repo, "commit", "-qm", "ignore scaffolding")
        wt = repo.parent / "wt-scaffold"
        _git(repo, "worktree", "add", "-q", "-b", "wb-scaffold", str(wt))
        (wt / "RALPH.md").write_text("# scaffolding\n")
        (wt / ".ralph-logs").mkdir()
        (wt / ".ralph-logs" / "run.log").write_text("log\n")
        GitAdapter(repo).remove_worktree(wt, "feature")
        assert not wt.exists()


class TestForceRemoveWorktree:
    def test_destroys_dirty_and_unlanded_work(self, repo: Path) -> None:
        """The explicitly-named destructive variant: discards what the
        fail-closed path refuses."""
        wt = _worktree_with_commit(repo, "wt-doomed")
        (wt / "dirty.py").write_text("also at risk\n")
        GitAdapter(repo).force_remove_worktree(wt)
        assert not wt.exists()

    @patch("milknado.adapters.git.subprocess.run")
    def test_passes_force_flag(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok()
        adapter.force_remove_worktree(Path("/tmp/wt"))
        mock_run.assert_called_once_with(
            ["git", "worktree", "remove", "--force", "/tmp/wt"],
            cwd=adapter._root,
            capture_output=True,
            text=True,
            check=True,
        )


class TestForceIsReachableOnlyViaDiscard:
    """Acceptance 3: `--force` teardown must be reachable only through the
    explicit discard path — a source-level audit of the production tree."""

    SRC = Path(__file__).parent.parent / "src" / "milknado"

    def _py_sources(self) -> list[Path]:
        return [p for p in self.SRC.rglob("*.py") if "_vendor" not in p.parts]

    def test_worktree_force_flag_lives_only_in_force_remove_worktree(self) -> None:
        offenders = []
        for path in self._py_sources():
            text = path.read_text()
            if "worktree" in text and '"remove", "--force"' in text:
                offenders.append(path)
        assert offenders == [self.SRC / "adapters" / "git.py"], (
            f"worktree remove --force found outside GitAdapter: {offenders}"
        )
        # And inside git.py it appears exactly once — in force_remove_worktree.
        text = (self.SRC / "adapters" / "git.py").read_text()
        assert text.count('"remove", "--force"') == 1
        body = text.split("def force_remove_worktree")[1]
        assert '"remove", "--force"' in body

    def test_force_remove_worktree_called_only_by_discard(self) -> None:
        callers = []
        for path in self._py_sources():
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if "force_remove_worktree(" in stripped and not stripped.startswith("def "):
                    callers.append((str(path.relative_to(self.SRC)), stripped))
        assert callers == [
            ("domains/execution/executor.py", "self._git.force_remove_worktree(wt_path)")
        ], f"force_remove_worktree must be called only by WorktreeManager.discard: {callers}"


class TestPruneWorktrees:
    @patch("milknado.adapters.git.subprocess.run")
    def test_calls_git_worktree_prune(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.return_value = _ok()
        adapter.prune_worktrees()
        mock_run.assert_called_once_with(
            ["git", "worktree", "prune"],
            cwd=adapter._root,
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("milknado.adapters.git.subprocess.run")
    def test_propagates_error(self, mock_run: MagicMock, adapter: GitAdapter) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        with pytest.raises(subprocess.CalledProcessError):
            adapter.prune_worktrees()


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
