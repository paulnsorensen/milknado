from __future__ import annotations

import pytest

from milknado.domains.batching import FileChange, NewRelationship, SymbolRef, plan_batches


def _batch_index(plan, change_id: str) -> int:
    """Return the batch index containing change_id."""
    for batch in plan.batches:
        if change_id in batch.change_ids:
            return batch.index
    raise KeyError(f"{change_id} not found in any batch")


def _all_change_ids(plan) -> set[str]:
    return {cid for batch in plan.batches for cid in batch.change_ids}


def test_empty_changes_returns_empty_optimal() -> None:
    plan = plan_batches([], budget=70_000)
    assert plan.batches == ()
    assert plan.spread_report == ()
    assert plan.solver_status == "OPTIMAL"


def test_duplicate_ids_raises() -> None:
    a = FileChange(id="1", path="a.py")
    b = FileChange(id="1", path="b.py")
    with pytest.raises(ValueError, match="duplicate"):
        plan_batches([a, b], budget=70_000)


def test_precedence_respected(tmp_path) -> None:
    a = FileChange(id="a", path="a.py", edit_kind="delete")
    b = FileChange(id="b", path="b.py", edit_kind="delete", depends_on=("a",))
    plan = plan_batches([a, b], budget=70_000, root=tmp_path)
    assert _batch_index(plan, "a") <= _batch_index(plan, "b")


def test_budget_forces_three_batches(tmp_path) -> None:
    changes = [FileChange(id=str(i), path=f"x{i}.py", edit_kind="add") for i in range(3)]
    plan = plan_batches(changes, budget=1900, root=tmp_path)  # each = 1875; 2 won't fit
    assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
    assert len(plan.batches) == 3
    assert _all_change_ids(plan) == {"0", "1", "2"}


def test_cycle_co_batches(tmp_path) -> None:
    a = FileChange(id="a", path="a.py", edit_kind="delete")
    b = FileChange(id="b", path="b.py", edit_kind="delete")
    rels = [
        NewRelationship(source_change_id="a", dependant_change_id="b", reason="new_import"),
        NewRelationship(source_change_id="b", dependant_change_id="a", reason="new_import"),
    ]
    plan = plan_batches([a, b], budget=70_000, new_relationships=rels, root=tmp_path)
    assert _batch_index(plan, "a") == _batch_index(plan, "b")


def test_same_symbol_co_locates(tmp_path) -> None:
    sym = SymbolRef(name="UserService", file="a.py")
    changes = [
        FileChange(id=str(i), path=f"f{i}.py", edit_kind="delete", symbols=(sym,))
        for i in range(5)
    ]
    plan = plan_batches(changes, budget=70_000, root=tmp_path)
    assert all(ss.spread == 0 for ss in plan.spread_report)


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
    actual_distance = abs(_batch_index(plan, "a") - _batch_index(plan, "b"))
    spread_entry = next(ss for ss in plan.spread_report if ss.symbol == sym)
    assert spread_entry.spread == actual_distance
    assert actual_distance == 1


def test_oversized_single_change_infeasible(tmp_path) -> None:
    c = FileChange(id="1", path="big.py", edit_kind="add")  # 1875 tokens
    plan = plan_batches([c], budget=100, root=tmp_path)
    assert plan.solver_status == "INFEASIBLE"


def test_timeout_returns_feasible_or_unknown(tmp_path) -> None:
    changes = [FileChange(id=str(i), path=f"f{i}.py", edit_kind="add") for i in range(30)]
    plan = plan_batches(changes, budget=2000, time_limit_s=0.05, root=tmp_path)
    assert plan.solver_status in ("OPTIMAL", "FEASIBLE", "UNKNOWN")


def test_batch_depends_on_chain(tmp_path) -> None:
    """Chain a -> b -> c produces sequential depends_on."""
    a = FileChange(id="a", path="a.py", edit_kind="delete")
    b = FileChange(id="b", path="b.py", edit_kind="delete", depends_on=("a",))
    c = FileChange(id="c", path="c.py", edit_kind="delete", depends_on=("b",))
    plan = plan_batches([a, b, c], budget=70_000, root=tmp_path)
    assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
    batch_a = next(bat for bat in plan.batches if "a" in bat.change_ids)
    batch_b = next(bat for bat in plan.batches if "b" in bat.change_ids)
    batch_c = next(bat for bat in plan.batches if "c" in bat.change_ids)
    # b depends on a's batch index
    assert batch_a.index in batch_b.depends_on or batch_a.index == batch_b.index
    # c depends on b's batch index
    assert batch_b.index in batch_c.depends_on or batch_b.index == batch_c.index


def test_new_relationship_adds_edge(tmp_path) -> None:
    a = FileChange(id="a", path="a.py", edit_kind="delete")
    b = FileChange(id="b", path="b.py", edit_kind="delete")
    rel = NewRelationship(source_change_id="a", dependant_change_id="b", reason="new_call")
    plan = plan_batches([a, b], budget=70_000, new_relationships=[rel], root=tmp_path)
    assert _batch_index(plan, "a") <= _batch_index(plan, "b")
