"""Dispatch frees a clean orphan worktree slot and keeps any orphan holding work."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from milknado.adapters import GitAdapter
from milknado.domains.execution.executor import WorktreeManager
from tests.execution_session_fixtures import init_repo

_BRANCH = "milknado/1-task"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def orphan(tmp_path: Path) -> tuple[Path, Path, WorktreeManager]:
    repo = init_repo(tmp_path)
    wt = repo / "milknado-wt-1-task"
    _ = _git(repo, "worktree", "add", "-q", "-b", _BRANCH, str(wt))
    return repo, wt, WorktreeManager(GitAdapter(repo))


def test_clean_landed_orphan_is_removed_with_its_branch(
    orphan: tuple[Path, Path, WorktreeManager],
) -> None:
    repo, wt, manager = orphan

    slot = manager.relocate_occupied(wt, _BRANCH, repo)

    assert slot == (wt, _BRANCH)
    assert not wt.exists()
    assert _BRANCH not in _git(repo, "branch", "--list")


def test_dirty_orphan_is_kept_and_dispatch_relocates(
    orphan: tuple[Path, Path, WorktreeManager],
) -> None:
    repo, wt, manager = orphan
    _ = (wt / "wip.txt").write_text("unsaved\n", encoding="utf-8")

    path, branch = manager.relocate_occupied(wt, _BRANCH, repo)

    assert (path, branch) != (wt, _BRANCH)
    assert (wt / "wip.txt").read_text(encoding="utf-8") == "unsaved\n"
    assert _BRANCH in _git(repo, "branch", "--list")


def test_orphan_with_unlanded_commit_is_kept_and_dispatch_relocates(
    orphan: tuple[Path, Path, WorktreeManager],
) -> None:
    repo, wt, manager = orphan
    _ = (wt / "work.txt").write_text("done\n", encoding="utf-8")
    _ = _git(wt, "add", "work.txt")
    _ = _git(wt, "commit", "-q", "-m", "unlanded work")

    path, branch = manager.relocate_occupied(wt, _BRANCH, repo)

    assert (path, branch) != (wt, _BRANCH)
    assert (wt / "work.txt").exists()
    assert "unlanded work" in _git(repo, "log", "-1", "--format=%s", _BRANCH)


def test_unremovable_path_is_left_alone(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    plain = repo / "milknado-wt-1-task"
    plain.mkdir()
    manager = WorktreeManager(GitAdapter(repo))

    path, _ = manager.relocate_occupied(plain, _BRANCH, repo)

    assert path != plain
    assert plain.exists()
