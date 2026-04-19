from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class MilknadoError(Exception):
    """Base for all milknado-specific exceptions."""


class RebaseAbortError(MilknadoError):
    def __init__(self, worktree: Path, stderr: str = "") -> None:
        self.worktree = worktree
        self.stderr = stderr
        super().__init__(f"git rebase --abort failed in {worktree}")


class RalphMarkdownWriteError(MilknadoError):
    def __init__(self, target: Path) -> None:
        self.target = target
        super().__init__(f"Failed to write RALPH.md to {target}")


class CompletionTimeout(MilknadoError):
    def __init__(self, active_run_ids: set[str]) -> None:
        self.active_run_ids = active_run_ids
        ids_str = ", ".join(sorted(active_run_ids))
        super().__init__(
            f"Timed out waiting for completion. Active runs: {ids_str}"
        )


class PlanningFailed(MilknadoError):
    def __init__(self, stderr: str) -> None:
        self.stderr = stderr
        super().__init__(
            f"Planning agent exited non-zero. stderr: {stderr[:200]}"
        )


class InvalidTransition(MilknadoError, ValueError):
    def __init__(
        self,
        node_id: int,
        current: Any,
        target: Any,
        valid_targets: set[Any],
    ) -> None:
        self.node_id = node_id
        self.current = current
        self.target = target
        self.valid_targets = valid_targets
        valid_str = ", ".join(sorted(v.value for v in valid_targets))
        super().__init__(
            f"Node {node_id}: cannot transition from {current.value} "
            f"to {target.value}. Valid: [{valid_str}]"
        )


class TransientDispatchError(MilknadoError):
    """Marker for errors that may be retried."""
