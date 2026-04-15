from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"


VALID_TRANSITIONS: dict[NodeStatus, set[NodeStatus]] = {
    NodeStatus.PENDING: {NodeStatus.RUNNING, NodeStatus.BLOCKED, NodeStatus.FAILED},
    NodeStatus.RUNNING: {
        NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.BLOCKED, NodeStatus.PENDING,
    },
    NodeStatus.BLOCKED: {NodeStatus.PENDING},
    NodeStatus.FAILED: {NodeStatus.PENDING},
    NodeStatus.DONE: set(),
}


@dataclass(frozen=True)
class MikadoNode:
    id: int
    description: str
    status: NodeStatus = NodeStatus.PENDING
    parent_id: int | None = None
    worktree_path: str | None = None
    branch_name: str | None = None
    run_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


@dataclass(frozen=True)
class MikadoEdge:
    parent_id: int
    child_id: int


@dataclass(frozen=True)
class RebaseResult:
    success: bool
    conflicting_files: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class CompletionEvent:
    run_id: str
    success: bool


@dataclass(frozen=True)
class FileOwnership:
    node_id: int
    file_path: str
