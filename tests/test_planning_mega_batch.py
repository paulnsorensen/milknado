from __future__ import annotations

import pytest

from milknado.domains.batching.change import Batch, BatchPlan
from milknado.domains.batching.solver import STATUS_OPTIMAL
from milknado.domains.common.errors import MegaBatchAborted
from milknado.domains.planning.planner import _check_mega_batch


def _plan(*batches: Batch) -> BatchPlan:
    return BatchPlan(
        batches=tuple(batches),
        spread_report=(),
        solver_status=STATUS_OPTIMAL,
    )


def _batch(*change_ids: str) -> Batch:
    return Batch(index=0, change_ids=change_ids, depends_on=())


class TestCheckMegaBatchNoRaise:
    def test_single_batch_at_threshold_does_not_raise(self) -> None:
        plan = _plan(_batch("a", "b", "c", "d", "e"))
        _check_mega_batch(plan, force_single_batch=False, threshold=5)

    def test_single_batch_below_threshold_does_not_raise(self) -> None:
        plan = _plan(_batch("a", "b"))
        _check_mega_batch(plan, force_single_batch=False, threshold=5)

    def test_empty_single_batch_does_not_raise(self) -> None:
        plan = _plan(_batch())
        _check_mega_batch(plan, force_single_batch=False, threshold=5)

    def test_multiple_batches_over_threshold_does_not_raise(self) -> None:
        b0 = Batch(index=0, change_ids=("a", "b", "c", "d", "e", "f"), depends_on=())
        b1 = Batch(index=1, change_ids=("g",), depends_on=(0,))
        plan = _plan(b0, b1)
        _check_mega_batch(plan, force_single_batch=False, threshold=5)

    def test_force_single_batch_bypasses_guard(self) -> None:
        plan = _plan(_batch("a", "b", "c", "d", "e", "f"))
        _check_mega_batch(plan, force_single_batch=True, threshold=5)


class TestCheckMegaBatchRaises:
    def test_single_batch_over_threshold_raises(self) -> None:
        plan = _plan(_batch("a", "b", "c", "d", "e", "f"))
        with pytest.raises(MegaBatchAborted) as exc_info:
            _check_mega_batch(plan, force_single_batch=False, threshold=5)
        assert exc_info.value.change_count == 6
        assert exc_info.value.threshold == 5

    def test_error_message_names_change_count(self) -> None:
        plan = _plan(_batch(*[f"c{i}" for i in range(10)]))
        with pytest.raises(MegaBatchAborted, match="10 changes"):
            _check_mega_batch(plan, force_single_batch=False, threshold=5)

    def test_error_message_mentions_force_flag(self) -> None:
        plan = _plan(_batch(*[f"c{i}" for i in range(10)]))
        with pytest.raises(MegaBatchAborted, match="--force-single-batch"):
            _check_mega_batch(plan, force_single_batch=False, threshold=5)

    def test_custom_threshold_respected(self) -> None:
        plan = _plan(_batch("a", "b", "c"))
        with pytest.raises(MegaBatchAborted) as exc_info:
            _check_mega_batch(plan, force_single_batch=False, threshold=2)
        assert exc_info.value.change_count == 3
        assert exc_info.value.threshold == 2
