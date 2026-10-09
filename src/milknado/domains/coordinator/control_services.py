from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from milknado.domains.common import SessionEvent, WorkerOwner
from milknado.domains.coordinator.recovery import RecoveryRuntime
from milknado.domains.graph import (
    ExecutionGroup,
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    TaskAttempt,
)
from milknado.domains.planning import Planner
from milknado.loop.sessions import RuntimeSession


class ReviewDecisionPort(Protocol):
    def __call__(
        self, request: GoalReviewDecisionRequest, *, decided_by: str
    ) -> GoalReviewRecord: ...


class TurnPreflightError(ValueError):
    """The provider turn failed before a worker started."""


@dataclass(frozen=True, slots=True)
class TurnRuntimeHooks:
    turn_id: str
    identity: Callable[[str], None]
    event: Callable[[SessionEvent], None]


@dataclass(frozen=True, slots=True)
class TurnIdentity:
    family: str
    session_id: str


@dataclass(frozen=True, slots=True)
class TurnRunResult:
    session_id: str | None
    terminal_confirmed: bool


@dataclass(frozen=True, slots=True)
class TurnRuntimeResult:
    run: TurnRunResult | None
    recovery_turn_confirmed: bool | None = None


@dataclass(frozen=True, slots=True)
class TurnRuntimeRequest:
    provider: str
    prompt: str
    group: ExecutionGroup | None
    identity: TurnIdentity | None
    hooks: TurnRuntimeHooks
    attempt: TaskAttempt | None = None


class TurnRuntimePort(Protocol):
    def run(self, request: TurnRuntimeRequest) -> TurnRuntimeResult: ...


@dataclass(frozen=True, slots=True)
class CoordinatorServices:
    planner: Planner | None = None
    runtime_session: Callable[[str], RuntimeSession | None] | None = None
    recovery_runtime: RecoveryRuntime | None = None
    review_decision: ReviewDecisionPort | None = None
    turn_runtime: TurnRuntimePort | None = None
    turn_owner: Callable[[str], WorkerOwner] | None = None
    turn_cancel: Callable[[str], bool] | None = None
