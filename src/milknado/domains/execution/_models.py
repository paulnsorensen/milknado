from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class NodeClaimRejected(ValueError):
    pass


class PreservedWorkerRun(Exception):
    def __init__(self, node_id: int, run_id: str, owner_run_id: str | None = None) -> None:
        self.node_id: int = node_id
        self.run_id: str = run_id
        self.owner_run_id: str | None = owner_run_id
        super().__init__(f"node {node_id} worker {run_id} did not confirm exit")


@dataclass(frozen=True)
class DispatchResult:
    node_id: int
    worktree: Path
    run_id: str


@dataclass(frozen=True)
class CompletionResult:
    node_id: int
    rebased: bool
    newly_ready: list[int]
    rebase_conflict: RebaseConflict | None = None
    redispatch: DispatchResult | None = None
    blocked: bool = False
    review_notification_failed: bool = False
    review_audit_failed: bool = False


@dataclass(frozen=True)
class RebaseConflict:
    node_id: int
    description: str
    conflicting_files: tuple[str, ...]
    detail: str
