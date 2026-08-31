from __future__ import annotations

from pathlib import Path

from milknado.domains.common.types import NodeStatus


class MilknadoError(Exception):
    """Base for all milknado-specific exceptions."""


class GitOperationError(MilknadoError):
    def __init__(self, operation: str, detail: str = "") -> None:
        self.operation: str = operation
        self.detail: str = detail
        message = f"git {operation} failed"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class RebaseAbortError(MilknadoError):
    def __init__(self, worktree: Path, stderr: str = "") -> None:
        self.worktree: Path = worktree
        self.stderr: str = stderr
        super().__init__(f"git rebase --abort failed in {worktree}")


class UnlandedWorkError(MilknadoError):
    """Refusal to remove a worktree that still holds dirty or unlanded work.

    Teardown is fail-closed: destroying work must be an explicitly named act
    (``GitAdapter.force_remove_worktree``, reachable only via
    ``WorktreeManager.discard``), never a default. ``at_risk`` names exactly
    what refusing preserved — dirty files and/or the unlanded commit range.
    """

    def __init__(self, worktree: Path, at_risk: str) -> None:
        self.worktree: Path = worktree
        self.at_risk: str = at_risk
        super().__init__(f"refusing to remove worktree {worktree}: {at_risk}")


class RalphMarkdownWriteError(MilknadoError):
    def __init__(self, path: Path, cause: OSError | None = None) -> None:
        self.path: Path = path
        self.cause: OSError | None = cause
        super().__init__(f"Failed to write RALPH.md to {path}")


class CompletionTimeout(MilknadoError):
    def __init__(self, active_run_ids: set[str], waited_seconds: float = 0.0) -> None:
        self.active_run_ids: set[str] = active_run_ids
        self.waited_seconds: float = waited_seconds
        ids_str = ", ".join(sorted(active_run_ids))
        super().__init__(
            f"Timed out after {waited_seconds:.1f}s waiting for completion. Active runs: {ids_str}"
        )


class InvalidTransition(MilknadoError, ValueError):
    def __init__(
        self,
        node_id: int,
        current: NodeStatus,
        target: NodeStatus,
        valid_targets: tuple[NodeStatus, ...],
    ) -> None:
        self.node_id: int = node_id
        self.current: NodeStatus = current
        self.target: NodeStatus = target
        self.valid_targets: tuple[NodeStatus, ...] = valid_targets
        valid_str = ", ".join(sorted(v.value for v in valid_targets))
        super().__init__(
            f"Node {node_id}: cannot transition from {current.value} "
            + f"to {target.value}. Valid: [{valid_str}]"
        )


class TransientDispatchError(MilknadoError):
    """Marker for errors that may be retried."""


class MegaBatchAborted(MilknadoError):
    def __init__(self, change_count: int, threshold: int) -> None:
        self.change_count: int = change_count
        self.threshold: int = threshold
        super().__init__(
            f"Mega-batch detected: {change_count} changes packed into a single batch "
            + f"(threshold={threshold}). Narrow scope or pass --force-single-batch."
        )


class ArchiveIneligible(MilknadoError, ValueError):
    """Refusal to archive nodes that are not DONE or block other work."""

    def __init__(self, node_ids: tuple[int, ...]) -> None:
        self.node_ids: tuple[int, ...] = node_ids
        ids_str = ", ".join(str(node_id) for node_id in node_ids)
        super().__init__(f"nodes not eligible for archive (non-DONE or blocking): {ids_str}")


class InvalidContainment(MilknadoError, ValueError):
    def __init__(self, parent_kind: object, child_kind: object) -> None:
        self.parent_kind: object = parent_kind
        self.child_kind: object = child_kind
        super().__init__(f"kind={child_kind!r} is not a valid child of kind={parent_kind!r}")
