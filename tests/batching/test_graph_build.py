from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from milknado.domains.batching.change import FileChange, NewRelationship, SymbolRef
from milknado.domains.batching.graph_build import (
    build_change_graph,
    contract_sccs,
    symbols_by_scc,
    validate_no_symbol_overlap,
)
from milknado.domains.batching.solver import plan_batches
from milknado.domains.common.protocols import CrgPort


@pytest.fixture()
def mock_crg() -> MagicMock:
    return MagicMock(spec=CrgPort)


def test_new_relationships_basic(mock_crg: MagicMock) -> None:
    a = FileChange(id="a", path="a.py")
    b = FileChange(id="b", path="b.py")
    rel = NewRelationship(source_change_id="a", dependant_change_id="b", reason="new_import")
    nodes, edges, _ = build_change_graph([a, b], new_relationships=[rel])
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


# ---------------------------------------------------------------------------
# Symbol-aware edge attribution (Commit 3)
# ---------------------------------------------------------------------------


def test_multi_change_per_path_distinct_symbols(mock_crg: MagicMock) -> None:
    """CRG edge with qualified source routes to correct change by symbol match."""
    sym_a = SymbolRef(name="a", file="src/foo.py")
    sym_x = SymbolRef(name="x", file="src/bar.py")
    change_a = FileChange(id="change_a", path="src/foo.py", symbols=(sym_a,))
    change_b = FileChange(
        id="change_b", path="src/foo.py", symbols=(SymbolRef(name="b", file="src/foo.py"),)
    )
    change_x = FileChange(id="change_x", path="src/bar.py", symbols=(sym_x,))
    mock_crg.get_impact_radius.return_value = {
        "edges": [{"src": "src/foo.py::a", "dst": "src/bar.py::x"}]
    }
    _, edges, _ = build_change_graph([change_a, change_b, change_x], crg=mock_crg)
    # Only change_a (symbol "a") should be attributed, not change_b (symbol "b")
    assert ("change_a", "change_x") in edges
    assert ("change_b", "change_x") not in edges


def test_multi_change_per_path_unknown_symbol_fans_out(mock_crg: MagicMock) -> None:
    """Path-only source with multiple candidates fans out to all candidate ids."""
    change_a = FileChange(
        id="change_a", path="src/foo.py", symbols=(SymbolRef(name="a", file="src/foo.py"),)
    )
    change_b = FileChange(
        id="change_b", path="src/foo.py", symbols=(SymbolRef(name="b", file="src/foo.py"),)
    )
    change_x = FileChange(
        id="change_x", path="src/bar.py", symbols=(SymbolRef(name="x", file="src/bar.py"),)
    )
    mock_crg.get_impact_radius.return_value = {
        "edges": [{"src": "src/foo.py", "dst": "src/bar.py::x"}]
    }
    _, edges, _ = build_change_graph([change_a, change_b, change_x], crg=mock_crg)
    assert ("change_a", "change_x") in edges
    assert ("change_b", "change_x") in edges


def test_multi_change_per_path_ambiguous_symbol_fans_out(mock_crg: MagicMock) -> None:
    """Qualified source naming a symbol not in any FileChange fans out to all candidates."""
    change_a = FileChange(
        id="change_a", path="src/foo.py", symbols=(SymbolRef(name="a", file="src/foo.py"),)
    )
    change_b = FileChange(
        id="change_b", path="src/foo.py", symbols=(SymbolRef(name="b", file="src/foo.py"),)
    )
    change_x = FileChange(id="change_x", path="src/bar.py")
    mock_crg.get_impact_radius.return_value = {
        "edges": [{"src": "src/foo.py::unknown_sym", "dst": "src/bar.py"}]
    }
    _, edges, _ = build_change_graph([change_a, change_b, change_x], crg=mock_crg)
    assert ("change_a", "change_x") in edges
    assert ("change_b", "change_x") in edges


def test_impacted_files_fans_out(mock_crg: MagicMock) -> None:
    """impacted_files path-only entries fan out to all changes on that path."""
    change_a = FileChange(id="change_a", path="src/foo.py")
    change_b = FileChange(id="change_b", path="src/foo.py")
    change_bar = FileChange(id="change_bar", path="src/bar.py")
    mock_crg.get_impact_radius.return_value = {"impacted_files": ["src/foo.py"]}
    _, edges, _ = build_change_graph([change_a, change_b, change_bar], crg=mock_crg)
    # change_bar is the queried change; both change_a and change_b are on "src/foo.py"
    assert ("change_bar", "change_a") in edges
    assert ("change_bar", "change_b") in edges


def test_overlapping_symbols_rejected() -> None:
    """Two FileChanges declaring the same symbol on the same path raises ValueError."""
    sym = SymbolRef(name="a", file="src/foo.py")
    change_a = FileChange(id="change_a", path="src/foo.py", symbols=(sym,))
    change_b = FileChange(id="change_b", path="src/foo.py", symbols=(sym,))
    with pytest.raises(ValueError, match="overlapping symbol"):
        plan_batches([change_a, change_b])


def test_cross_file_symbol_ref_no_overlap() -> None:
    """Same symbol name on different files is not an overlap — each change owns its own path."""
    sym_a = SymbolRef(name="shared", file="src/foo.py")
    sym_b = SymbolRef(name="shared", file="src/bar.py")
    change_1 = FileChange(id="change_1", path="src/foo.py", symbols=(sym_a,))
    change_2 = FileChange(id="change_2", path="src/bar.py", symbols=(sym_b,))
    # Different files — must not raise.
    validate_no_symbol_overlap([change_1, change_2])
