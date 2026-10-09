from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupWorkspace:
    worktree_path: str
    branch_name: str
    provider_session_id: str


@dataclass(frozen=True)
class ExecutionGroup:
    id: str
    graph_id: str
    worktree_path: str
    branch_name: str
    provider_session_id: str
    source_group_id: str | None = None


@dataclass(frozen=True)
class TaskOutcome:
    status: str
    result: str


@dataclass(frozen=True)
class TaskAttempt:
    group_id: str
    node_id: int
    run_id: str
    attempt_id: str


@dataclass(frozen=True)
class GroupPlan:
    graph_id: str
    tasks: tuple[int, ...]
    workspace: GroupWorkspace
    source_group_id: str | None = None
