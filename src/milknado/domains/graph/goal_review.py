"""Goal-level review and execution admission contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from milknado.domains.graph.commands import CommandReceipt


class GoalReviewDecision(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class GoalReviewRequest:
    goal_id: int
    goal_revision: str
    evidence: str
    proposed_change: str
    affected_node_ids: tuple[int, ...] | None = None
    reviewer: str = ""
    assessed_at: str | None = None


@dataclass(frozen=True, slots=True)
class GoalReviewDecisionRequest:
    review_id: int
    decision: GoalReviewDecision
    decided_at: str | None = None


@dataclass(frozen=True, slots=True)
class GoalReviewRecord:
    review_id: int
    goal_id: int
    goal_revision: str
    evidence: str
    proposed_change: str
    decision: GoalReviewDecision
    affected_node_ids: tuple[int, ...] | None
    reviewer: str
    assessed_at: str
    decided_at: str | None
    decided_by: str | None
    interruption_receipts: tuple[CommandReceipt, ...] = ()

    @property
    def unbounded(self) -> bool:
        return self.affected_node_ids is None


@dataclass(frozen=True, slots=True)
class GoalAdmission:
    allowed: bool
    goal_id: int | None
    review_id: int | None
    decision: GoalReviewDecision | None
    affected_node_ids: tuple[int, ...] | None
    reason: str | None = None


class GoalReviewSubjectError(ValueError):
    """Raised when review approval addresses anything but an execution goal."""


class GoalAdmissionDenied(ValueError):
    """Raised while pending goal evidence pauses execution."""

    admission: GoalAdmission

    def __init__(self, admission: GoalAdmission) -> None:
        self.admission = admission
        super().__init__(admission.reason or "execution admission denied")
