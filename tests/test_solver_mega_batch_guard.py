"""US-005: _check_mega_batch guard — acceptance criteria coverage.

Four scenarios:
  1. 1×6 batch aborts at default threshold (5).
  2. 1×4 batch passes (below threshold).
  3. 2×N batches pass (guard is batch-count-aware, not total-change-count-aware).
  4. --force-single-batch override passes regardless of size.
"""

from __future__ import annotations

import pytest

from milknado.domains.batching.change import Batch, BatchPlan
from milknado.domains.batching.solver import STATUS_OPTIMAL
from milknado.domains.common.errors import MegaBatchAborted
from milknado.domains.planning.planner import _check_mega_batch

_DEFAULT_THRESHOLD = 5


def _make_plan(*batches: Batch) -> BatchPlan:
    return BatchPlan(batches=tuple(batches), spread_report=(), solver_status=STATUS_OPTIMAL)


def _make_batch(index: int, *change_ids: str, depends_on: tuple[int, ...] = ()) -> Batch:
    return Batch(index=index, change_ids=change_ids, depends_on=depends_on)


class TestSingleBatchSixChangesAborts:
    """1×6 batch → exceeds default threshold of 5 → MegaBatchAborted."""

    def test_raises_mega_batch_aborted(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d", "e", "f"))
        with pytest.raises(MegaBatchAborted):
            _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)

    def test_error_carries_change_count(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d", "e", "f"))
        with pytest.raises(MegaBatchAborted) as exc_info:
            _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)
        assert exc_info.value.change_count == 6

    def test_error_carries_threshold(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d", "e", "f"))
        with pytest.raises(MegaBatchAborted) as exc_info:
            _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)
        assert exc_info.value.threshold == _DEFAULT_THRESHOLD

    def test_error_message_names_change_count(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d", "e", "f"))
        with pytest.raises(MegaBatchAborted, match="6"):
            _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)

    def test_error_message_recommends_force_flag(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d", "e", "f"))
        with pytest.raises(MegaBatchAborted, match="--force-single-batch"):
            _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)


class TestSingleBatchFourChangesPasses:
    """1×4 batch → below default threshold → no exception."""

    def test_does_not_raise(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d"))
        _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)

    def test_exactly_at_threshold_does_not_raise(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d", "e"))
        _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)


class TestTwoBatchesPassRegardlessOfSize:
    """2×N batches → guard only fires for a single batch → no exception."""

    def test_two_large_batches_do_not_raise(self) -> None:
        b0 = _make_batch(0, "a", "b", "c", "d", "e", "f")
        b1 = _make_batch(1, "g", "h", "i", "j", "k", "l", depends_on=(0,))
        plan = _make_plan(b0, b1)
        _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)

    def test_two_small_batches_do_not_raise(self) -> None:
        b0 = _make_batch(0, "a", "b")
        b1 = _make_batch(1, "c", "d", depends_on=(0,))
        plan = _make_plan(b0, b1)
        _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)

    def test_empty_plan_does_not_raise(self) -> None:
        plan = _make_plan()
        _check_mega_batch(plan, force_single_batch=False, threshold=_DEFAULT_THRESHOLD)


class TestForceSingleBatchOverride:
    """--force-single-batch=True bypasses the guard regardless of change count."""

    def test_oversized_single_batch_passes_with_force(self) -> None:
        plan = _make_plan(_make_batch(0, *[f"c{i}" for i in range(20)]))
        _check_mega_batch(plan, force_single_batch=True, threshold=_DEFAULT_THRESHOLD)

    def test_force_with_exactly_threshold_plus_one(self) -> None:
        plan = _make_plan(_make_batch(0, "a", "b", "c", "d", "e", "f"))
        _check_mega_batch(plan, force_single_batch=True, threshold=_DEFAULT_THRESHOLD)
