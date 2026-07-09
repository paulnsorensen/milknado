from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"


class NodeKind(Enum):
    ROADMAP = "roadmap"
    GOAL = "goal"
    TASK = "task"


class WorktreeMode(Enum):
    ISOLATE = "isolate"
    THIS_BRANCH = "this_branch"


# Flavor is a free string validated against BUILTIN_FLAVORS ∪ TOML-declared
# names at write paths (see config.MilknadoConfig.flavors); the registry
# mechanism lives in milknado, the vocabulary lives in user space.
BUILTIN_FLAVORS: frozenset[str] = frozenset(
    {"implement", "spec", "spike", "prototype", "research"}
)
DEFAULT_FLAVOR = "implement"


VALID_TRANSITIONS: dict[NodeStatus, set[NodeStatus]] = {
    NodeStatus.PENDING: {NodeStatus.RUNNING, NodeStatus.BLOCKED, NodeStatus.FAILED},
    NodeStatus.RUNNING: {
        NodeStatus.DONE,
        NodeStatus.FAILED,
        NodeStatus.BLOCKED,
        NodeStatus.PENDING,
    },
    NodeStatus.BLOCKED: {NodeStatus.PENDING},
    NodeStatus.FAILED: {NodeStatus.PENDING},
    NodeStatus.DONE: set(),
}

# Containment hierarchy: which child kinds are legal under each parent kind.
# Root nodes (parent_id=None) accept any kind — bare task lists stay legal.
VALID_CHILD_KINDS: dict[NodeKind, set[NodeKind]] = {
    NodeKind.ROADMAP: {NodeKind.GOAL},
    NodeKind.GOAL: {NodeKind.GOAL, NodeKind.TASK},
    NodeKind.TASK: {NodeKind.TASK},
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
    pid: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    dispatched_at: datetime | None = None
    oversized: bool = False
    batch_index: int | None = None
    completion_duration_seconds: float | None = None
    kind: NodeKind = NodeKind.TASK
    flavor: str | None = None
    goal_run_id: str | None = None  # coordinator run that claimed this goal (goal_claims table)
    # deterministic wiki key (uuid5); set for imported roadmap/goal nodes
    wiki_ref: str | None = None
    # opaque repo-relative markdown reference (wiki_ref precedent); no existence enforcement
    artifact_path: str | None = None

    def __post_init__(self) -> None:
        if self.kind == NodeKind.TASK:
            # Default to "implement" when None; any explicit value is valid.
            if self.flavor is None:
                object.__setattr__(self, "flavor", DEFAULT_FLAVOR)
        else:
            if self.flavor is not None:  # pragma: no cover — graph.add_node enforces this first
                raise ValueError(
                    f"flavor must be None for kind={self.kind.value}; got {self.flavor!r}"
                )


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
class TilthMap:
    scope: Path
    budget_tokens: int
    data: dict[str, Any]


@dataclass(frozen=True)
class DegradationMarker:
    source: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class NodeSpec:
    """Optional add_node attributes, grouped to keep add_node's signature small."""

    kind: NodeKind = NodeKind.TASK
    flavor: str | None = None
    wiki_ref: str | None = None
    artifact_path: str | None = None
    prereqs: tuple[int, ...] = ()
    flavor_registry: frozenset[str] = BUILTIN_FLAVORS
    oversized: bool = False
    batch_index: int | None = None


@dataclass(frozen=True)
class RunResult:
    """Terminal outcome of a run, threaded from graph.finish_run to persistence."""

    status: str
    exit_code: int | None
    timed_out: bool
    ended_at: str
    error: str | None = None
    detail: str | None = None
    rebased: bool | None = None
