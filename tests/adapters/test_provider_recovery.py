from __future__ import annotations

import subprocess
from pathlib import Path

from milknado.adapters import DeferredProviderRecovery, ExistingWorktreeRecovery
from milknado.domains.coordinator.recovery import ProviderIdentity
from milknado.domains.graph import ExecutionGroup


def test_restart_does_not_claim_provider_resume_without_a_turn(tmp_path: Path) -> None:
    port = DeferredProviderRecovery()
    assert port.recover(ProviderIdentity("claude", "provider-1"), tmp_path) == "unavailable"
    assert port.recover(ProviderIdentity("codex", "thread-1"), tmp_path) == "unavailable"


def test_existing_worktree_requires_same_repository_and_branch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _ = subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    _ = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "--allow-empty",
            "-qm",
            "seed",
        ],
        check=True,
    )
    worktree = tmp_path / "worker"
    _ = subprocess.run(
        ["git", "-C", str(root), "worktree", "add", "-qb", "task", str(worktree)],
        check=True,
    )
    group = ExecutionGroup("group", "graph", str(worktree), "task", "provider-1")
    port = ExistingWorktreeRecovery(root)
    assert port.restore(group)
    assert not port.restore(ExecutionGroup("group", "graph", str(worktree), "other", "provider-1"))
    assert not port.restore(ExecutionGroup("group", "graph", str(tmp_path), "task", "provider-1"))
