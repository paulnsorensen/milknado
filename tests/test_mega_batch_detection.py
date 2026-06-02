"""Mega-batch detection — BatchPlan.mega_batch_change_count acceptance coverage.

Detection is domain-owned: BatchPlan scans ALL batches and reports the largest
change count exceeding MEGA_BATCH_THRESHOLD (or None). This replaces the old
single-batch-only check_mega_batch guard, whose early-return on len(batches) != 1
silently passed multi-batch plans that held one oversized batch.
"""

from __future__ import annotations

from milknado.domains.batching.change import MEGA_BATCH_THRESHOLD, Batch, BatchPlan
from milknado.domains.batching.solver import STATUS_OPTIMAL


def _make_plan(*batches: Batch) -> BatchPlan:
    return BatchPlan(batches=tuple(batches), spread_report=(), solver_status=STATUS_OPTIMAL)


def _make_batch(index: int, *change_ids: str, depends_on: tuple[int, ...] = ()) -> Batch:
    return Batch(index=index, change_ids=change_ids, depends_on=depends_on)


class TestSingleBatch:
    """A single batch is flagged purely on its own change count."""

    def test_six_changes_reports_six(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d", "e", "f"))
        assert plan.mega_batch_change_count == 6

    def test_exactly_at_threshold_reports_none(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d", "e"))
        assert plan.mega_batch_change_count is None

    def test_below_threshold_reports_none(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d"))
        assert plan.mega_batch_change_count is None


class TestMultiBatchRegression:
    """The bug fix: a multi-batch plan with one oversized batch must be flagged.

    The old guard early-returned on len(batches) != 1, so [3, 6] returned None.
    """

    def test_one_oversized_batch_among_several_is_flagged(self) -> None:
        plan = _make_plan(
            _make_batch(0, "a", "b", "c"),
            _make_batch(1, "d", "e", "f", "g", "h", "i", depends_on=(0,)),
        )
        assert plan.mega_batch_change_count == 6

    def test_reports_largest_offending_batch(self) -> None:
        plan = _make_plan(
            _make_batch(0, *[f"x{i}" for i in range(6)]),
            _make_batch(1, *[f"y{i}" for i in range(12)], depends_on=(0,)),
        )
        assert plan.mega_batch_change_count == 12

    def test_all_batches_at_or_below_threshold_reports_none(self) -> None:
        plan = _make_plan(
            _make_batch(0, "a", "b", "c", "d", "e"),
            _make_batch(1, "f", "g", depends_on=(0,)),
        )
        assert plan.mega_batch_change_count is None


class TestEmptyPlan:
    def test_empty_plan_reports_none(self) -> None:
        assert _make_plan().mega_batch_change_count is None


def test_threshold_constant_is_five() -> None:
    assert MEGA_BATCH_THRESHOLD == 5
