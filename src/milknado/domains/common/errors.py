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
    def __init__(self, path: Path, cause: OSError | None = None) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"Failed to write RALPH.md to {path}")


class CompletionTimeout(MilknadoError):
    def __init__(self, active_run_ids: set[str], waited_seconds: float = 0.0) -> None:
        self.active_run_ids = active_run_ids
        self.waited_seconds = waited_seconds
        ids_str = ", ".join(sorted(active_run_ids))
        super().__init__(
            f"Timed out after {waited_seconds:.1f}s waiting for completion. Active runs: {ids_str}"
        )


class PlanningFailed(MilknadoError):
    def __init__(self, exit_code: int, stderr: str) -> None:
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(
            f"Planning agent exited {exit_code}. stderr: {stderr[:200]}"
        )


class InvalidTransition(MilknadoError, ValueError):
    def __init__(
        self,
        node_id: int,
        current: Any,
        target: Any,
        valid_targets: tuple[Any, ...],
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
