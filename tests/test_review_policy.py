from __future__ import annotations

import pytest

from milknado.domains.execution._review_policy import ReviewPolicy, decide_review


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (("approve", True, 0, 1, "block"), ("allow_merge", "approved", 0)),
        (("approve", False, 0, 1, "warn"), ("block", "approval_audit_failed", 0)),
        (("reject", False, 0, 1, "warn"), ("block", "rejection_audit_failed", 1)),
        (("error", True, 0, 1, "warn"), ("block", "reviewer_error", 0)),
        (("reject", True, 0, 1, "warn"), ("redispatch", "rejected", 1)),
        (("reject", True, 1, 1, "block"), ("block", "round_limit", 2)),
        (("reject", True, 1, 1, "warn"), ("allow_merge", "round_limit_warn", 2)),
    ],
)
def test_review_decision_table(
    case: tuple[str, bool, int, int, str], expected: tuple[str, str, int]
) -> None:
    verdict, audited, round_number, limit, on_reject = case
    decision = decide_review(verdict, audited, round_number, ReviewPolicy(limit, on_reject))
    assert (decision.action, decision.reason, decision.next_round) == expected
