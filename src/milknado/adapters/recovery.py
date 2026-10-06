from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from milknado.adapters.git import GitAdapter
from milknado.domains.coordinator import ProviderIdentity, RecoveryOutcome
from milknado.domains.graph import ExecutionGroup


class DeferredProviderRecovery:
    def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome:
        del identity, cwd
        return "unavailable"


@dataclass(frozen=True, slots=True)
class ExistingWorktreeRecovery:
    root: Path

    def restore(self, group: ExecutionGroup) -> bool:
        path = Path(group.worktree_path)
        if not path.is_absolute() or not path.is_dir():
            return False
        git = GitAdapter(self.root)
        common = git.git_common_dir(path)
        if common is None or common != git.git_common_dir(self.root):
            return False
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel", "--abbrev-ref", "HEAD"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and result.stdout.splitlines() == [
            str(path.resolve()),
            group.branch_name,
        ]
