from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from milknado.domains.batching.change import FileChange, SymbolRef
from milknado.domains.batching.graph_build import (
    build_change_graph,
    contract_sccs,
    symbols_by_scc,
)
from milknado.domains.common.protocols import CrgPort


@pytest.fixture()
def mock_crg() -> MagicMock:
    return MagicMock(spec=CrgPort)


def test_extra_edges_basic(mock_crg: MagicMock) -> None:
    a = FileChange(id="a", path="a.py")
    b = FileChange(id="b", path="b.py")
    nodes, edges, _ = build_change_graph([a, b], extra_edges=[("a", "b")])
    assert ("a", "b") in edges


def test_depends_on_emits_edge() -> None:
    a = FileChange(id="a", path="a.py")
    b = FileChange(id="b", path="b.py", depends_on=("a",))
    nodes, edges, _ = build_change_graph([a, b])
    assert ("a", "b") in edges


def test_crg_impacted_files_key(mock_crg: MagicMock) -> None:
    mock_crg.get_impact_radius.return_value = {"impacted_files": ["b.py"]}
    a = FileChange(id="a", path="a.py")
    b = FileChange(id="b", path="b.py")
    _, edges, _ = build_change_graph([a, b], crg=mock_crg)
    assert ("a", "b") in edges


def test_crg_edges_key(mock_crg: MagicMock) -> None:
    mock_crg.get_impact_radius.return_value = {"edges": [{"src": "a.py", "dst": "b.py"}]}
    a = FileChange(id="a", path="a.py")
    b = FileChange(id="b", path="b.py")
    _, edges, _ = build_change_graph([a, b], crg=mock_crg)
    assert ("a", "b") in edges


def test_unknown_crg_path_dropped(mock_crg: MagicMock) -> None:
    mock_crg.get_impact_radius.return_value = {"impacted_files": ["unknown.py"]}
    a = FileChange(id="a", path="a.py")
    _, edges, _ = build_change_graph([a], crg=mock_crg)
    assert edges == ()


def test_unknown_dep_id_raises() -> None:
    a = FileChange(id="a", path="a.py", depends_on=("nonexistent",))
    with pytest.raises(ValueError, match="unknown depends_on id"):
        build_change_graph([a])


def test_scc_linear_dag() -> None:
    nodes = ("a", "b", "c")
    edges = (("a", "b"), ("b", "c"))
    scc_of, dag_edges = contract_sccs(nodes, edges)
    # Each node in its own SCC
    assert len(set(scc_of.values())) == 3
    assert len(dag_edges) == 2


def test_scc_two_cycle() -> None:
    nodes = ("a", "b")
    edges = (("a", "b"), ("b", "a"))
    scc_of, dag_edges = contract_sccs(nodes, edges)
    # a and b in same SCC
    assert scc_of["a"] == scc_of["b"]
    assert dag_edges == ()


def test_scc_three_cycle() -> None:
    nodes = ("a", "b", "c")
    edges = (("a", "b"), ("b", "c"), ("c", "a"))
    scc_of, dag_edges = contract_sccs(nodes, edges)
    assert scc_of["a"] == scc_of["b"] == scc_of["c"]
    assert dag_edges == ()


def test_scc_mixed() -> None:
    # a->b cycle, c is separate, c->a
    nodes = ("a", "b", "c")
    edges = (("a", "b"), ("b", "a"), ("c", "a"))
    scc_of, dag_edges = contract_sccs(nodes, edges)
    assert scc_of["a"] == scc_of["b"]
    assert scc_of["c"] != scc_of["a"]
    assert len(dag_edges) == 1


def test_symbols_by_scc_union() -> None:
    sym_a = SymbolRef(name="Foo", file="a.py")
    sym_b = SymbolRef(name="Bar", file="b.py")
    scc_of = {"a": "a", "b": "a", "c": "c"}
    sym_by_node: dict[str, tuple[SymbolRef, ...]] = {"a": (sym_a,), "b": (sym_b,), "c": ()}
    result = symbols_by_scc(scc_of, sym_by_node)
    assert set(result["a"]) == {sym_a, sym_b}
    assert result.get("c", ()) == ()
