"""Isolated worktree slot selection refuses exhausted or escaping retry slots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import GitPort
from milknado.domains.common.errors import GitOperationError
from milknado.domains.dispatch.isolate import create_isolated_worktree

_PATTERN = "milknado-{node_id}-{slug}"


class _Git:
    branches_taken: bool

    def __init__(self, *, branches_taken: bool) -> None:
        self.branches_taken = branches_taken
        self.created: list[Path] = []

    def current_branch(self) -> str:
        return "main"

    def resolve_ref(self, ref: str) -> str:
        _ = ref
        return "base"

    def branch_exists(self, branch: str) -> bool:
        _ = branch
        return self.branches_taken

    def create_worktree(self, path: Path, _branch: str) -> None:
        self.created.append(path)
        path.mkdir(parents=True)


def test_every_slot_with_an_existing_branch_is_reported_occupied(tmp_path: Path) -> None:
    git = _Git(branches_taken=True)

    with pytest.raises(GitOperationError, match="all 100 are occupied"):
        _ = create_isolated_worktree(
            cast(GitPort, cast(object, git)), tmp_path, 3, "busy", _PATTERN
        )

    assert git.created == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_retry_slot_that_resolves_outside_the_project_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "elsewhere"
    root.mkdir()
    outside.mkdir()
    (root / "milknado-4-escape").mkdir()
    (root / "milknado-4-escape-retry2").symlink_to(outside, target_is_directory=True)
    git = _Git(branches_taken=False)

    with pytest.raises(ValueError, match="resolves outside project_root"):
        _ = create_isolated_worktree(cast(GitPort, cast(object, git)), root, 4, "escape", _PATTERN)

    assert git.created == []
    assert list(outside.iterdir()) == []
