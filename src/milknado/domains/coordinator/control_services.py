from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from milknado.domains.coordinator.recovery import RecoveryRuntime
from milknado.domains.graph import GoalReviewDecisionRequest, GoalReviewRecord
from milknado.domains.planning import Planner
from milknado.loop.sessions import RuntimeSession


class ReviewDecisionPort(Protocol):
    def __call__(
        self, request: GoalReviewDecisionRequest, *, decided_by: str
    ) -> GoalReviewRecord: ...


@dataclass(frozen=True, slots=True)
class CoordinatorServices:
    planner: Planner | None = None
    runtime_session: Callable[[str], RuntimeSession | None] | None = None
    recovery_runtime: RecoveryRuntime | None = None
    review_decision: ReviewDecisionPort | None = None
