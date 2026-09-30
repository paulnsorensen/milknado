from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ReviewPolicy:
    max_rounds: int
    on_reject: str


@dataclass(frozen=True)
class ReviewDecision:
    action: Literal["allow_merge", "redispatch", "block"]
    reason: str
    next_round: int


def decide_review(
    verdict: str,
    audit_succeeded: bool,
    current_round: int,
    policy: ReviewPolicy,
) -> ReviewDecision:
    next_round = current_round + (verdict == "reject")
    if not audit_succeeded:
        reason = "approval_audit_failed" if verdict == "approve" else "rejection_audit_failed"
        return ReviewDecision("block", reason, next_round)
    if verdict == "error":
        return ReviewDecision("block", "reviewer_error", next_round)
    if verdict == "approve":
        return ReviewDecision("allow_merge", "approved", next_round)
    if current_round < policy.max_rounds:
        return ReviewDecision("redispatch", "rejected", next_round)
    if policy.on_reject == "block":
        return ReviewDecision("block", "round_limit", next_round)
    return ReviewDecision("allow_merge", "round_limit_warn", next_round)
