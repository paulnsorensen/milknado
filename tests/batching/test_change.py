from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from milknado.domains.batching import BatchPlan, FileChange, NewRelationship, SymbolRef
from milknado.domains.batching.change import Batch, SymbolSpread


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


def test_new_relationship_fields() -> None:
    rel = NewRelationship(
        source_change_id="a",
        dependant_change_id="b",
        reason="new_import",
    )
    assert rel.source_change_id == "a"
    assert rel.dependant_change_id == "b"
    assert rel.reason == "new_import"


def test_new_relationship_is_frozen() -> None:
    rel = NewRelationship(source_change_id="a", dependant_change_id="b", reason="new_file")
    with pytest.raises(FrozenInstanceError):
        rel.reason = "new_call"  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_batch_fields() -> None:
    b = Batch(index=0, change_ids=("a", "b"), depends_on=())
    assert b.index == 0
    assert b.change_ids == ("a", "b")
    assert b.depends_on == ()
    assert b.oversized is False


def test_batch_oversized_flag() -> None:
    b = Batch(index=1, change_ids=("x",), depends_on=(0,), oversized=True)
    assert b.oversized is True
    assert b.depends_on == (0,)


def test_batch_is_frozen() -> None:
    b = Batch(index=0, change_ids=("a",), depends_on=())
    with pytest.raises(FrozenInstanceError):
        b.index = 1  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_symbol_spread_fields() -> None:
    sym = SymbolRef(name="foo", file="bar.py")
    ss = SymbolSpread(symbol=sym, spread=2)
    assert ss.symbol == sym
    assert ss.spread == 2


def test_symbol_spread_is_frozen() -> None:
    sym = SymbolRef(name="foo", file="bar.py")
    ss = SymbolSpread(symbol=sym, spread=1)
    with pytest.raises(FrozenInstanceError):
        ss.spread = 0  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def test_batch_plan_fields() -> None:
    sym = SymbolRef(name="x", file="f.py")
    batch = Batch(index=0, change_ids=("a",), depends_on=())
    ss = SymbolSpread(symbol=sym, spread=0)
    p = BatchPlan(batches=(batch,), spread_report=(ss,), solver_status="OPTIMAL")
    assert p.batches == (batch,)
    assert p.spread_report == (ss,)
    assert p.solver_status == "OPTIMAL"


def test_batch_plan_is_frozen() -> None:
    p = BatchPlan(batches=(), spread_report=(), solver_status="OPTIMAL")
    with pytest.raises(FrozenInstanceError):
        p.solver_status = "INFEASIBLE"  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
