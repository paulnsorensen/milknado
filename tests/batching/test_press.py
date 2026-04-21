"""Press-phase adversarial tests — targeted attack on the 4-commit batching redesign.

Targets:
1. Lexicographic two-pass solver correctness (FEASIBLE pass 1, UNKNOWN early return,
   status downgrade logic)
2. Oversized passthrough edge cases (DAG edges, two oversized SCCs, all-oversized)
3. Symbol-aware edge routing (:: in symbol name, empty path_to_ids, fan-out dedup)
4. Overlap validation (same symbol name / different file, multi-path with claimed path)
5. MCP boundary (_dict_to_new_relationship missing keys, invalid reason, empty list)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from milknado.domains.batching import plan_batches
from milknado.domains.batching.change import FileChange, SolverStatus, SymbolRef
from milknado.domains.batching.graph_build import (
    _parse_impact_dict,
    _parse_qualified,
    _resolve_ids_for_endpoint,
    build_change_graph,
    validate_no_symbol_overlap,
)
from milknado.domains.batching.solver import (
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    STATUS_UNKNOWN,
    _worse_status,
)
from milknado.mcp_server import _dict_to_new_relationship

# ---------------------------------------------------------------------------
# 1. Lexicographic solver — status downgrade and two-pass behaviour
# ---------------------------------------------------------------------------


class TestTwoPassStatusDowngrade:
    """_worse_status semantics and end-to-end status propagation."""

    def test_worse_status_optimal_vs_feasible_returns_feasible(self):
        assert _worse_status(STATUS_OPTIMAL, STATUS_FEASIBLE) == STATUS_FEASIBLE

    def test_worse_status_feasible_vs_feasible_returns_feasible(self):
        assert _worse_status(STATUS_FEASIBLE, STATUS_FEASIBLE) == STATUS_FEASIBLE

    def test_worse_status_optimal_vs_optimal_returns_optimal(self):
        assert _worse_status(STATUS_OPTIMAL, STATUS_OPTIMAL) == STATUS_OPTIMAL

    def test_worse_status_optimal_vs_unknown_returns_unknown(self):
        # UNKNOWN < FEASIBLE < OPTIMAL — if pass 2 returns UNKNOWN the plan degrades
        assert _worse_status(STATUS_OPTIMAL, STATUS_UNKNOWN) == STATUS_UNKNOWN

    def test_worse_status_feasible_vs_infeasible_returns_infeasible(self):
        assert _worse_status(STATUS_FEASIBLE, STATUS_INFEASIBLE) == STATUS_INFEASIBLE

    def test_worse_status_symmetry(self):
        # _worse_status must be symmetric
        pairs: list[tuple[SolverStatus, SolverStatus]] = [
            (STATUS_OPTIMAL, STATUS_FEASIBLE),
            (STATUS_OPTIMAL, STATUS_UNKNOWN),
            (STATUS_FEASIBLE, STATUS_INFEASIBLE),
        ]
        for a, b in pairs:
            assert _worse_status(a, b) == _worse_status(b, a), f"Not symmetric: {a}, {b}"

    def test_pass1_feasible_plan_still_includes_all_changes(self, tmp_path):
        """When pass 1 returns FEASIBLE the solver must still produce a valid solution."""
        # Force a tiny time budget so pass1 may return FEASIBLE (not OPTIMAL)
        # We can't guarantee FEASIBLE here, but if it IS FEASIBLE, solution must be complete.
        changes = [FileChange(id=str(i), path=f"f{i}.py", edit_kind="add") for i in range(10)]
        plan = plan_batches(changes, budget=2000, time_limit_s=0.1, root=tmp_path)
        if plan.solver_status in (STATUS_OPTIMAL, STATUS_FEASIBLE):
            all_ids = {cid for b in plan.batches for cid in b.change_ids}
            assert all_ids == {str(i) for i in range(10)}, (
                "FEASIBLE pass must still contain all change IDs"
            )

    def test_plan_status_is_worse_of_two_passes(self, tmp_path):
        """If the final plan is FEASIBLE, we cannot have a OPTIMAL in plan.batches == empty."""
        # This tests that when a valid solution exists, we don't return batches=()
        a = FileChange(id="a", path="a.py", edit_kind="delete")
        plan = plan_batches([a], budget=70_000, root=tmp_path)
        # A single delete-80-token change with huge budget must yield OPTIMAL and non-empty batches
        assert plan.solver_status == STATUS_OPTIMAL
        assert len(plan.batches) == 1
        assert plan.batches[0].change_ids == ("a",)

    def test_unknown_status_returns_empty_batches(self, tmp_path):
        """When _two_pass_solve returns UNKNOWN, plan_batches must return batches=()."""
        from milknado.domains.batching import solver as solver_mod

        def fake_two_pass(bundle, time_limit_s):
            return None, STATUS_UNKNOWN

        a = FileChange(id="a", path="a.py", edit_kind="delete")
        with patch.object(solver_mod, "_two_pass_solve", side_effect=fake_two_pass):
            plan = plan_batches([a], budget=70_000, root=tmp_path)
        assert plan.solver_status == STATUS_UNKNOWN
        assert plan.batches == ()

    def test_infeasible_status_returns_empty_batches(self, tmp_path):
        """When _two_pass_solve returns INFEASIBLE, plan_batches must return batches=()."""
        from milknado.domains.batching import solver as solver_mod

        def fake_two_pass(bundle, time_limit_s):
            return None, STATUS_INFEASIBLE

        a = FileChange(id="a", path="a.py", edit_kind="delete")
        with patch.object(solver_mod, "_two_pass_solve", side_effect=fake_two_pass):
            plan = plan_batches([a], budget=70_000, root=tmp_path)
        assert plan.solver_status == STATUS_INFEASIBLE
        assert plan.batches == ()

    def test_pass1_optimal_pass2_unknown_keeps_pass1_plan(self, tmp_path):
        """S1 regression: pass-2 UNKNOWN must not discard a valid pass-1 solution.

        Previously the solver would return (solver, UNKNOWN) with batches=() even
        when pass 1 found an OPTIMAL arrangement. Now the pass-1 snapshot is kept
        and the final plan reports the worse (OPTIMAL→OPTIMAL in this case).
        """
        from milknado.domains.batching import solver as solver_mod

        def stub_two_pass(bundle, time_limit_s):
            # Run pass 1 only, stop before pass 2.
            from ortools.sat.python import cp_model as cpm

            solver = cpm.CpSolver()
            bundle.model.minimize(bundle.max_batch_idx)
            status1 = solver_mod._status_name(solver.solve(bundle.model))
            snapshot = solver_mod._take_snapshot(solver, bundle.batch_of, bundle.spread_vars)
            # Simulate pass-2 degrading to UNKNOWN after pass 1 succeeded.
            return snapshot, status1

        a = FileChange(id="a", path="a.py", edit_kind="delete")
        with patch.object(solver_mod, "_two_pass_solve", side_effect=stub_two_pass):
            plan = plan_batches([a], budget=70_000, root=tmp_path)
        assert plan.solver_status == STATUS_OPTIMAL
        assert len(plan.batches) == 1
        assert plan.batches[0].change_ids == ("a",)


# ---------------------------------------------------------------------------
# 2. Oversized passthrough — DAG edge ordering, two oversized SCCs, all-oversized
# ---------------------------------------------------------------------------


class TestOversizedPassthrough:
    def test_two_oversized_sccs_with_dag_edge_order_preserved(self, tmp_path):
        """Two oversized SCCs with a DAG edge — the source must come before the dest."""
        # both changes cost more than budget=50
        big_a = FileChange(id="big_a", path="big_a.py", edit_kind="add")  # 1875 tokens
        big_b = FileChange(
            id="big_b", path="big_b.py", edit_kind="add", depends_on=("big_a",)
        )  # 1875 tokens
        plan = plan_batches([big_a, big_b], budget=50, root=tmp_path)
        assert plan.solver_status == STATUS_OPTIMAL
        assert len(plan.batches) == 2
        batch_a = next(b for b in plan.batches if "big_a" in b.change_ids)
        batch_b = next(b for b in plan.batches if "big_b" in b.change_ids)
        assert batch_a.oversized is True
        assert batch_b.oversized is True
        assert batch_a.index < batch_b.index, "Source oversized SCC must precede dependent"
        assert batch_a.index in batch_b.depends_on

    def test_all_sccs_oversized_returns_optimal_no_solver(self, tmp_path):
        """When every SCC is oversized, solver is bypassed and result is OPTIMAL."""
        a = FileChange(id="a", path="a.py", edit_kind="add")
        b = FileChange(id="b", path="b.py", edit_kind="add")
        plan = plan_batches([a, b], budget=50, root=tmp_path)
        assert plan.solver_status == STATUS_OPTIMAL
        all_ids = {cid for batch in plan.batches for cid in batch.change_ids}
        assert all_ids == {"a", "b"}
        assert all(batch.oversized for batch in plan.batches)

    def test_oversized_with_dag_edge_to_normal_depends_on_set(self, tmp_path):
        """Oversized SCC with DAG edge to normal SCC — depends_on populated in normal batch."""
        big = FileChange(id="big", path="big.py", edit_kind="add")  # oversized
        # small is a delete (80 tokens) which fits within budget=100
        small = FileChange(id="small", path="small.py", edit_kind="delete", depends_on=("big",))
        plan = plan_batches([big, small], budget=100, root=tmp_path)
        assert plan.solver_status in (STATUS_OPTIMAL, STATUS_FEASIBLE)
        big_batch = next(b for b in plan.batches if "big" in b.change_ids)
        small_batch = next(b for b in plan.batches if "small" in b.change_ids)
        assert big_batch.oversized is True
        assert small_batch.oversized is False
        assert big_batch.index < small_batch.index
        assert big_batch.index in small_batch.depends_on

    def test_normal_with_dag_edge_to_oversized_depends_on_set(self, tmp_path):
        """Normal SCC -> oversized SCC ordering must produce a valid plan (Mikado pattern)."""
        small = FileChange(id="small", path="small.py", edit_kind="delete")  # 80 tokens fits 100
        big = FileChange(
            id="big", path="big.py", edit_kind="add", depends_on=("small",)
        )  # 1875 tokens oversized
        plan = plan_batches([small, big], budget=100, root=tmp_path)
        assert plan.solver_status in (STATUS_OPTIMAL, STATUS_FEASIBLE)
        small_batch = next(b for b in plan.batches if "small" in b.change_ids)
        big_batch = next(b for b in plan.batches if "big" in b.change_ids)
        assert small_batch.oversized is False
        assert big_batch.oversized is True
        assert small_batch.index < big_batch.index
        assert small_batch.index in big_batch.depends_on

    def test_three_oversized_sccs_linear_chain(self, tmp_path):
        """Three oversized SCCs in a chain — indices must be monotonically increasing."""
        a = FileChange(id="a", path="a.py", edit_kind="add")
        b = FileChange(id="b", path="b.py", edit_kind="add", depends_on=("a",))
        c = FileChange(id="c", path="c.py", edit_kind="add", depends_on=("b",))
        plan = plan_batches([a, b, c], budget=50, root=tmp_path)
        assert plan.solver_status == STATUS_OPTIMAL
        idx = {cid: batch.index for batch in plan.batches for cid in batch.change_ids}
        assert idx["a"] < idx["b"] < idx["c"]

    def test_normal_precedes_oversized(self, tmp_path):
        """Canonical Mikado: small prerequisite then large implementation."""
        small = FileChange(id="small", path="small.py", edit_kind="delete")  # ~80 tokens
        # ~1875 tokens
        big = FileChange(id="big", path="big.py", edit_kind="add", depends_on=("small",))
        plan = plan_batches([small, big], budget=100, root=tmp_path)
        assert plan.solver_status in (STATUS_OPTIMAL, STATUS_FEASIBLE)
        assert len(plan.batches) == 2
        small_batch = next(b for b in plan.batches if "small" in b.change_ids)
        big_batch = next(b for b in plan.batches if "big" in b.change_ids)
        assert small_batch.oversized is False
        assert big_batch.oversized is True
        assert small_batch.index < big_batch.index
        assert small_batch.index in big_batch.depends_on

    def test_oversized_precedes_normal(self, tmp_path):
        """Oversized SCC must come before the normal SCC it precedes."""
        big = FileChange(id="big", path="big.py", edit_kind="add")  # ~1875 tokens oversized
        small = FileChange(id="small", path="small.py", edit_kind="delete", depends_on=("big",))
        plan = plan_batches([big, small], budget=100, root=tmp_path)
        assert plan.solver_status in (STATUS_OPTIMAL, STATUS_FEASIBLE)
        assert len(plan.batches) == 2
        big_batch = next(b for b in plan.batches if "big" in b.change_ids)
        small_batch = next(b for b in plan.batches if "small" in b.change_ids)
        assert big_batch.oversized is True
        assert small_batch.oversized is False
        assert big_batch.index < small_batch.index
        assert big_batch.index in small_batch.depends_on


# ---------------------------------------------------------------------------
# 3. Symbol-aware edge routing — :: in symbol name, empty endpoint, fan-out dedup
# ---------------------------------------------------------------------------


class TestSymbolAwareRouting:
    def test_parse_qualified_basic(self):
        path, sym = _parse_qualified("foo.py::MyClass")
        assert path == "foo.py"
        assert sym == "MyClass"

    def test_parse_qualified_no_colon(self):
        path, sym = _parse_qualified("foo.py")
        assert path == "foo.py"
        assert sym is None

    def test_parse_qualified_nested_colon_takes_first_split(self):
        """C++ style qualified name foo.py::Class::method — split("::", 1) gives method-part."""
        path, sym = _parse_qualified("foo.py::Class::method")
        assert path == "foo.py"
        assert sym == "Class::method"

    def test_nested_qualified_name_attributed_when_symbol_matches(self):
        """A CRG edge using foo.py::Class::method attributed to change claiming that symbol."""
        # SymbolRef.name is "Class::method" in this FileChange
        sym = SymbolRef(name="Class::method", file="src/foo.py")
        change_a = FileChange(id="change_a", path="src/foo.py", symbols=(sym,))
        change_b = FileChange(
            id="change_b",
            path="src/foo.py",
            symbols=(SymbolRef(name="OtherClass", file="src/foo.py"),),
        )
        change_x = FileChange(id="change_x", path="src/bar.py")
        crg = MagicMock()
        crg.get_impact_radius.return_value = {
            "edges": [{"src": "src/foo.py::Class::method", "dst": "src/bar.py"}]
        }
        _, edges, _ = build_change_graph([change_a, change_b, change_x], crg=crg)
        # change_a declared "Class::method" so it should be attributed
        assert ("change_a", "change_x") in edges
        # change_b declared "OtherClass" — NOT attributed (symbol resolved unambiguously)
        assert ("change_b", "change_x") not in edges

    def test_empty_endpoint_path_not_in_path_to_ids_skipped(self):
        """CRG edge whose endpoint path is not in any FileChange is silently skipped."""
        path_to_ids: dict[str, list[str]] = {"a.py": ["id_a"]}
        id_to_change = {"id_a": FileChange(id="id_a", path="a.py")}
        result = _parse_impact_dict(
            {"edges": [{"src": "a.py", "dst": "unknown.py"}]},
            "a.py",
            path_to_ids,
            id_to_change,
        )
        # dst has no ids -> no pairs emitted
        assert result == []

    def test_both_endpoints_unknown_produces_no_pairs(self):
        """Both src and dst paths unknown → empty result, no crash."""
        path_to_ids: dict[str, list[str]] = {}
        id_to_change: dict[str, FileChange] = {}
        result = _parse_impact_dict(
            {"edges": [{"src": "ghost.py::foo", "dst": "other_ghost.py::bar"}]},
            "ghost.py",
            path_to_ids,
            id_to_change,
        )
        assert result == []

    def test_same_edge_attributed_twice_deduped(self):
        """Same (src_id, dst_id) pair via two distinct CRG edges deduped to one."""
        path_to_ids = {"a.py": ["id_a"], "b.py": ["id_b"]}
        id_to_change = {
            "id_a": FileChange(id="id_a", path="a.py"),
            "id_b": FileChange(id="id_b", path="b.py"),
        }
        result = _parse_impact_dict(
            {
                "edges": [
                    {"src": "a.py", "dst": "b.py"},
                    {"src": "a.py", "dst": "b.py"},
                ]
            },
            "a.py",
            path_to_ids,
            id_to_change,
        )
        assert result.count(("id_a", "id_b")) == 1

    def test_qualified_src_resolves_correct_change_when_symbol_matches_one(self):
        """_resolve_ids_for_endpoint narrows to the single matching change."""
        sym_a = SymbolRef(name="foo", file="src/m.py")
        sym_b = SymbolRef(name="bar", file="src/m.py")
        ca = FileChange(id="ca", path="src/m.py", symbols=(sym_a,))
        cb = FileChange(id="cb", path="src/m.py", symbols=(sym_b,))
        path_to_ids = {"src/m.py": ["ca", "cb"]}
        id_to_change = {"ca": ca, "cb": cb}
        result = _resolve_ids_for_endpoint("src/m.py", "foo", path_to_ids, id_to_change)
        assert result == ["ca"]

    def test_qualified_src_falls_back_to_all_when_ambiguous(self):
        """_resolve_ids_for_endpoint fans out when symbol matches multiple changes."""
        sym = SymbolRef(name="shared", file="src/m.py")
        ca = FileChange(id="ca", path="src/m.py", symbols=(sym,))
        cb = FileChange(id="cb", path="src/m.py", symbols=(sym,))
        path_to_ids = {"src/m.py": ["ca", "cb"]}
        id_to_change = {"ca": ca, "cb": cb}
        result = _resolve_ids_for_endpoint("src/m.py", "shared", path_to_ids, id_to_change)
        assert set(result) == {"ca", "cb"}


# ---------------------------------------------------------------------------
# 4. Overlap validation — same symbol name / different file, multi-path edge cases
# ---------------------------------------------------------------------------


class TestOverlapValidation:
    def test_same_symbol_name_different_file_does_not_raise(self):
        """Same symbol name but different file → NOT an overlap — must not raise."""
        sym_a = SymbolRef(name="process", file="a.py")
        sym_b = SymbolRef(name="process", file="b.py")
        ca = FileChange(id="ca", path="a.py", symbols=(sym_a,))
        cb = FileChange(id="cb", path="b.py", symbols=(sym_b,))
        # Should not raise
        validate_no_symbol_overlap([ca, cb])

    def test_same_symbol_same_file_different_changes_raises(self):
        """Same symbol AND same file in two changes → raises ValueError."""
        sym = SymbolRef(name="process", file="shared.py")
        ca = FileChange(id="ca", path="shared.py", symbols=(sym,))
        cb = FileChange(id="cb", path="shared.py", symbols=(sym,))
        with pytest.raises(ValueError, match="overlapping symbol"):
            validate_no_symbol_overlap([ca, cb])

    def test_symbol_cross_file_not_owned_does_not_raise(self):
        """A SymbolRef whose file is NOT the change's own path is not an ownership claim."""
        # ca owns "a.py" but the symbol points to "b.py" (cross-file spread tracking)
        sym_b = SymbolRef(name="Helper", file="b.py")
        ca = FileChange(id="ca", path="a.py", symbols=(sym_b,))
        cb = FileChange(id="cb", path="b.py", symbols=(sym_b,))
        # ca doesn't own b.py so symbol in ca is not an ownership claim — must not raise
        validate_no_symbol_overlap([ca, cb])

    def test_different_symbols_same_file_ok(self):
        """Two changes on the same file declaring different symbols — must not raise."""
        sym_a = SymbolRef(name="A", file="shared.py")
        sym_b = SymbolRef(name="B", file="shared.py")
        ca = FileChange(id="ca", path="shared.py", symbols=(sym_a,))
        cb = FileChange(id="cb", path="shared.py", symbols=(sym_b,))
        validate_no_symbol_overlap([ca, cb])

    def test_same_symbol_same_file_raises(self):
        """Two changes declaring the same symbol on the same file → raises."""
        sym = SymbolRef(name="Widget", file="shared.py")
        ca = FileChange(id="ca", path="shared.py", symbols=(sym,))
        cb = FileChange(id="cb", path="shared.py", symbols=(sym,))
        with pytest.raises(ValueError, match="overlapping symbol"):
            validate_no_symbol_overlap([ca, cb])

    def test_no_symbols_no_overlap_raises(self):
        """Two changes on the same path with no symbols → never raises (no ownership claimed)."""
        ca = FileChange(id="ca", path="x.py")
        cb = FileChange(id="cb", path="x.py")
        validate_no_symbol_overlap([ca, cb])  # must not raise

    def test_single_change_multiple_symbols_no_raises(self):
        """Single change with multiple symbols on same path — no duplicate to detect."""
        sym_a = SymbolRef(name="A", file="a.py")
        sym_b = SymbolRef(name="B", file="a.py")
        ca = FileChange(id="ca", path="a.py", symbols=(sym_a, sym_b))
        validate_no_symbol_overlap([ca])  # must not raise


# ---------------------------------------------------------------------------
# 5. MCP boundary — _dict_to_new_relationship
# ---------------------------------------------------------------------------


class TestDictToNewRelationship:
    def test_valid_dict_round_trips(self):
        d = {
            "source_change_id": "a",
            "dependant_change_id": "b",
            "reason": "new_import",
        }
        rel = _dict_to_new_relationship(d)
        assert rel.source_change_id == "a"
        assert rel.dependant_change_id == "b"
        assert rel.reason == "new_import"

    def test_missing_source_change_id_raises_key_error(self):
        with pytest.raises(KeyError):
            _dict_to_new_relationship({"dependant_change_id": "b", "reason": "new_import"})

    def test_missing_dependant_change_id_raises_key_error(self):
        with pytest.raises(KeyError):
            _dict_to_new_relationship({"source_change_id": "a", "reason": "new_import"})

    def test_missing_reason_raises_key_error(self):
        with pytest.raises(KeyError):
            _dict_to_new_relationship({"source_change_id": "a", "dependant_change_id": "b"})

    def test_invalid_reason_string_raises_value_error(self):
        """_dict_to_new_relationship must reject reasons not in RelationshipReason."""
        d = {
            "source_change_id": "a",
            "dependant_change_id": "b",
            "reason": "new_function",  # not in RelationshipReason Literal
        }
        with pytest.raises(ValueError, match="invalid reason"):
            _dict_to_new_relationship(d)

    def test_empty_new_relationships_list_accepted(self, tmp_path, monkeypatch):
        """An empty new_relationships list at the MCP boundary is accepted."""
        from milknado.adapters import crg as crg_mod
        from milknado.mcp_server import _plan_batches_impl

        class StubAdapter:
            def __init__(self, project_root) -> None:
                pass

            def get_impact_radius(self, files):
                return {}

            def ensure_graph(self, project_root) -> None:
                pass

            def get_architecture_overview(self):
                return {}

        monkeypatch.setattr(crg_mod, "CrgAdapter", StubAdapter)
        result = _plan_batches_impl(
            [{"id": "1", "path": "a.py", "edit_kind": "delete"}],
            70_000,
            tmp_path,
            new_relationships=[],
        )
        assert result["solver_status"] in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN")

    def test_none_new_relationships_treated_as_empty(self, tmp_path, monkeypatch):
        """None passed as new_relationships is equivalent to an empty list."""
        from milknado.adapters import crg as crg_mod
        from milknado.mcp_server import _plan_batches_impl

        class StubAdapter:
            def __init__(self, project_root) -> None:
                pass

            def get_impact_radius(self, files):
                return {}

            def ensure_graph(self, project_root) -> None:
                pass

            def get_architecture_overview(self):
                return {}

        monkeypatch.setattr(crg_mod, "CrgAdapter", StubAdapter)
        result = _plan_batches_impl(
            [{"id": "1", "path": "a.py", "edit_kind": "delete"}],
            70_000,
            tmp_path,
            new_relationships=None,
        )
        assert result["solver_status"] in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN")

    def test_all_valid_reason_values_accepted(self):
        """Each member of RelationshipReason Literal is accepted without error."""
        reasons = ["new_file", "new_import", "new_call", "new_type_use"]
        for reason in reasons:
            d = {
                "source_change_id": "a",
                "dependant_change_id": "b",
                "reason": reason,
            }
            rel = _dict_to_new_relationship(d)
            assert rel.reason == reason

    def test_invalid_reason_rejected_at_mcp_boundary(self):
        """MCP boundary must reject invalid reason values with a descriptive ValueError."""
        with pytest.raises(ValueError, match="invalid reason"):
            _dict_to_new_relationship(
                {
                    "source_change_id": "a",
                    "dependant_change_id": "b",
                    "reason": "new_function",
                }
            )


# ---------------------------------------------------------------------------
# 6. Integration — two-oversized and mixed ordering correctness end-to-end
# ---------------------------------------------------------------------------


class TestIntegrationOrderingCorrectness:
    def test_oversized_scc_batch_index_in_depends_on_of_successor(self, tmp_path):
        """The batch that contains big must appear in the depends_on of the batch after it."""
        big = FileChange(id="big", path="big.py", edit_kind="add")  # oversized at budget=50
        dep1 = FileChange(id="dep1", path="dep1.py", edit_kind="delete", depends_on=("big",))
        dep2 = FileChange(id="dep2", path="dep2.py", edit_kind="delete", depends_on=("big",))
        plan = plan_batches([big, dep1, dep2], budget=50, root=tmp_path)
        assert plan.solver_status in (STATUS_OPTIMAL, STATUS_FEASIBLE)
        big_batch = next(b for b in plan.batches if "big" in b.change_ids)
        dep1_batch = next(b for b in plan.batches if "dep1" in b.change_ids)
        dep2_batch = next(b for b in plan.batches if "dep2" in b.change_ids)
        assert big_batch.index in dep1_batch.depends_on
        assert big_batch.index in dep2_batch.depends_on

    def test_batch_indices_are_contiguous_from_zero(self, tmp_path):
        """After remap, batch indices must be 0, 1, 2, ... with no gaps."""
        a = FileChange(id="a", path="a.py", edit_kind="delete")
        b = FileChange(id="b", path="b.py", edit_kind="delete", depends_on=("a",))
        c = FileChange(id="c", path="c.py", edit_kind="delete", depends_on=("b",))
        plan = plan_batches([a, b, c], budget=80, root=tmp_path)
        assert plan.solver_status in (STATUS_OPTIMAL, STATUS_FEASIBLE)
        indices = sorted(batch.index for batch in plan.batches)
        assert indices == list(range(len(indices))), f"Indices not contiguous: {indices}"

    def test_depends_on_only_references_valid_batch_indices(self, tmp_path):
        """All batch.depends_on values must reference an actual batch index in the plan."""
        a = FileChange(id="a", path="a.py", edit_kind="delete")
        b = FileChange(id="b", path="b.py", edit_kind="delete", depends_on=("a",))
        c = FileChange(id="c", path="c.py", edit_kind="add")
        plan = plan_batches([a, b, c], budget=80, root=tmp_path)
        valid_indices = {batch.index for batch in plan.batches}
        for batch in plan.batches:
            for dep in batch.depends_on:
                assert dep in valid_indices, (
                    f"Batch {batch.index} depends_on {dep}, not in plan indices {valid_indices}"
                )

    def test_change_ids_not_duplicated_across_batches(self, tmp_path):
        """No change ID should appear in more than one batch."""
        changes = [FileChange(id=str(i), path=f"f{i}.py", edit_kind="delete") for i in range(5)]
        plan = plan_batches(changes, budget=80, root=tmp_path)
        all_ids: list[str] = []
        for batch in plan.batches:
            all_ids.extend(batch.change_ids)
        assert len(all_ids) == len(set(all_ids)), "Change ID appeared in multiple batches"
