"""Immutable requests and records for one logical task attempt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from milknado.domains.common import NodeStatus, RunResult


@dataclass(frozen=True, slots=True)
class StartTaskAttempt:
    node_id: int
    attempt_id: str
    owner_run_id: str
    invocation_id: str | None = None
    max_node_attempts: int = 1


StartState = Literal["unknown", "released", "executed", "startup_failed"]
AttemptOutcome = Literal[
    "active",
    "done",
    "cancelled",
    "review_blocked",
    "preserved",
    "startup_failed",
    "failed",
    "reset",
]


@dataclass(frozen=True, slots=True)
class TaskAttemptRecord:
    node_id: int
    attempt_id: str
    owner_run_id: str
    active_invocation_id: str
    max_node_attempts: int
    start_state: StartState
    outcome: AttemptOutcome
    failed_count: int | None
    disposition: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class FinalizeTaskFailure:
    node_id: int
    attempt_id: str
    owner_run_id: str
    run_id: str
    result: RunResult
    group_id: str | None = None
    group_result: str | None = None


@dataclass(frozen=True, slots=True)
class FailureDisposition:
    status: NodeStatus | Literal["lost_fence", "unknown_start", "startup_failed"]
    failed_count: int
    reason: str
