from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from milknado.domains.batching import BatchPlan, FileChange, SymbolRef


def test_symbol_ref_is_frozen() -> None:
    s = SymbolRef(name="foo", file="bar.py")
    with pytest.raises(FrozenInstanceError):
        s.name = "baz"  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_file_change_defaults() -> None:
    c = FileChange(id="1", path="a.py")
    assert c.edit_kind == "modify"
    assert c.symbols == ()
    assert c.depends_on == ()


def test_file_change_identity_by_id() -> None:
    a = FileChange(id="1", path="a.py")
    b = FileChange(id="1", path="a.py")
    assert a == b


def test_batch_plan_fields() -> None:
    p = BatchPlan(batches=(("a",),), spread_report={"x": 0}, solver_status="OPTIMAL")
    assert p.batches == (("a",),)
    assert p.spread_report == {"x": 0}
    assert p.solver_status == "OPTIMAL"
