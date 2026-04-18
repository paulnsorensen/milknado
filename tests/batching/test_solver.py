from __future__ import annotations

import pytest

from milknado.domains.batching import FileChange, SymbolRef, plan_batches


def test_empty_changes_returns_empty_optimal() -> None:
    plan = plan_batches([], budget=70_000)
    assert plan.batches == ()
    assert plan.spread_report == {}
    assert plan.solver_status == "OPTIMAL"


def test_duplicate_ids_raises() -> None:
    a = FileChange(id="1", path="a.py")
    b = FileChange(id="1", path="b.py")
    with pytest.raises(ValueError, match="duplicate"):
        plan_batches([a, b], budget=70_000)


def test_precedence_respected(tmp_path) -> None:
    a = FileChange(id="a", path="a.py", edit_kind="delete")  # 80 tokens
    b = FileChange(id="b", path="b.py", edit_kind="delete", depends_on=("a",))
    plan = plan_batches([a, b], budget=70_000, root=tmp_path)
    idx = {c: i for i, batch in enumerate(plan.batches) for c in batch}
    assert idx["a"] <= idx["b"]


def test_budget_forces_three_batches(tmp_path) -> None:
    changes = [FileChange(id=str(i), path=f"x{i}.py", edit_kind="add") for i in range(3)]
    plan = plan_batches(changes, budget=1900, root=tmp_path)  # each = 1875; 2 won't fit
    assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
    assert len(plan.batches) == 3
    assert {c for batch in plan.batches for c in batch} == {"0", "1", "2"}


def test_cycle_co_batches(tmp_path) -> None:
    a = FileChange(id="a", path="a.py", edit_kind="delete")
    b = FileChange(id="b", path="b.py", edit_kind="delete")
    plan = plan_batches([a, b], budget=70_000, extra_edges=[("a", "b"), ("b", "a")], root=tmp_path)
    idx = {c: i for i, batch in enumerate(plan.batches) for c in batch}
    assert idx["a"] == idx["b"]


def test_same_symbol_co_locates(tmp_path) -> None:
    sym = SymbolRef(name="UserService", file="a.py")
    changes = [
        FileChange(id=str(i), path=f"f{i}.py", edit_kind="delete", symbols=(sym,))
        for i in range(5)
    ]
    plan = plan_batches(changes, budget=70_000, root=tmp_path)
    assert all(v == 0 for v in plan.spread_report.values())


def test_spread_report_measures_distance_not_max_batch(tmp_path) -> None:
    """Regression: earlier version computed spread = max(batch_of), not max - min.
    Force a symbol across two batches via budget; spread must be 1, not 2+.
    """
    sym = SymbolRef(name="Service", file="svc.py")
    a = FileChange(id="a", path="a.py", edit_kind="add", symbols=(sym,))
    b = FileChange(id="b", path="b.py", edit_kind="add", symbols=(sym,))
    # each add-py is 1875 tokens; budget 2000 forces them into separate batches.
    plan = plan_batches([a, b], budget=2000, root=tmp_path)
    assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
    idx = {c: i for i, batch in enumerate(plan.batches) for c in batch}
    actual_distance = abs(idx["a"] - idx["b"])
    assert plan.spread_report["svc.py:Service"] == actual_distance
    assert actual_distance == 1


def test_oversized_single_change_infeasible(tmp_path) -> None:
    c = FileChange(id="1", path="big.py", edit_kind="add")  # 1875 tokens
    plan = plan_batches([c], budget=100, root=tmp_path)
    assert plan.solver_status == "INFEASIBLE"


def test_timeout_returns_feasible_or_unknown(tmp_path) -> None:
    changes = [FileChange(id=str(i), path=f"f{i}.py", edit_kind="add") for i in range(30)]
    plan = plan_batches(changes, budget=2000, time_limit_s=0.05, root=tmp_path)
    assert plan.solver_status in ("OPTIMAL", "FEASIBLE", "UNKNOWN")
