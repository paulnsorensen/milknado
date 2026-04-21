"""Adversarial / chaos tests for the batching slice.

Guilt-until-proven-innocent: every function is fragile until these pass.
Attack order: invalid inputs -> edge cases -> integration paths -> happy path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from milknado.domains.batching import plan_batches
from milknado.domains.batching.change import FileChange, NewRelationship, SymbolRef
from milknado.domains.batching.graph_build import (
    _parse_impact_dict,
    build_change_graph,
    contract_sccs,
)
from milknado.domains.batching.weights import _extension, estimate_tokens

# ---------------------------------------------------------------------------
# weights.py — _extension edge cases
# ---------------------------------------------------------------------------


class TestExtension:
    def test_dotfile_no_extension(self):
        assert _extension(".gitignore") == ""

    def test_dotfile_with_extension(self):
        # '.env.local' base starts with '.' -> empty
        assert _extension(".env.local") == ""

    def test_double_extension(self):
        assert _extension("a.py.bak") == "bak"

    def test_no_extension(self):
        assert _extension("no-ext") == ""

    def test_empty_path(self):
        assert _extension("") == ""

    def test_just_dot_only(self):
        # 'a.' -> base='a.', has dot, doesn't start with dot -> rsplit('.', 1)[-1] = ''
        result = _extension("a.")
        assert result == ""

    def test_weird_dots(self):
        assert _extension("weird.name.with.dots") == "dots"

    def test_windows_backslash_path(self):
        # rsplit('/') gives whole string as base; dot not leading
        ext = _extension("dir\\sub\\file.py")
        assert ext == "py"

    def test_path_with_emoji(self):
        ext = _extension("src/module_cheese.py")
        assert ext == "py"

    def test_extension_case_lowered(self):
        assert _extension("file.PY") == "py"
        assert _extension("file.Py") == "py"


# ---------------------------------------------------------------------------
# weights.py — estimate_tokens edge cases
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_zero_length_file_returns_zero(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        c = FileChange(id="1", path="empty.py", edit_kind="modify")
        from unittest.mock import patch as _patch

        with _patch("milknado.domains.batching.weights._tiktoken_count", return_value=0):
            result = estimate_tokens(c, tmp_path)
        assert result == 0

    def test_binary_file_does_not_crash(self, tmp_path):
        f = tmp_path / "binary.py"
        f.write_bytes(bytes(range(256)))
        c = FileChange(id="1", path="binary.py", edit_kind="modify")
        from unittest.mock import patch as _patch

        with _patch("milknado.domains.batching.weights._tiktoken_count", return_value=42):
            result = estimate_tokens(c, tmp_path)
        assert isinstance(result, int)
        assert result >= 0

    def test_path_traversal_falls_back_gracefully(self, tmp_path):
        # '../secret.py' resolves outside tmp_path; file missing -> heuristic fallback
        c = FileChange(id="1", path="../secret.py", edit_kind="modify")
        result = estimate_tokens(c, tmp_path)
        assert isinstance(result, int)
        assert result > 0

    def test_add_unknown_extension_uses_defaults(self, tmp_path):
        # DEFAULT_LINES=150, DEFAULT_TPL=8, HEADROOM=1.25 -> 1500
        c = FileChange(id="1", path="config", edit_kind="add")
        result = estimate_tokens(c, tmp_path)
        assert result == 1500

    def test_double_extension_bak_unknown(self, tmp_path):
        # ext='bak' not in tables -> DEFAULT
        c = FileChange(id="1", path="a.py.bak", edit_kind="add")
        result = estimate_tokens(c, tmp_path)
        assert result == 1500

    def test_invalid_edit_kind_does_not_crash(self, tmp_path):
        # edit_kind='garbage' not in FLAT_COST, not 'modify' -> falls to add path
        c = FileChange(id="1", path="a.py", edit_kind="garbage")  # type: ignore
        result = estimate_tokens(c, tmp_path)
        assert isinstance(result, int)
        assert result > 0


# ---------------------------------------------------------------------------
# graph_build.py — _parse_impact_dict chaos
# ---------------------------------------------------------------------------



class TestParseImpactDict:
    _ca = FileChange(id="id_a", path="a.py")
    _cb = FileChange(id="id_b", path="b.py")
    _path_to_ids = {"a.py": ["id_a"], "b.py": ["id_b"]}
    _id_to_change = {"id_a": _ca, "id_b": _cb}

    def test_edges_none_no_pairs(self):
        result = _parse_impact_dict({"edges": None}, "a.py", self._path_to_ids, self._id_to_change)
        assert result == []

    def test_edges_string_no_pairs(self):
        result = _parse_impact_dict(
            {"edges": "string"}, "a.py", self._path_to_ids, self._id_to_change
        )
        assert result == []

    def test_edges_list_of_empty_dicts_skipped(self):
        result = _parse_impact_dict(
            {"edges": [{}, {}]}, "a.py", self._path_to_ids, self._id_to_change
        )
        assert result == []

    def test_edges_missing_src_skipped(self):
        result = _parse_impact_dict(
            {"edges": [{"dst": "b.py"}]}, "a.py", self._path_to_ids, self._id_to_change
        )
        assert result == []

    def test_edges_missing_dst_skipped(self):
        result = _parse_impact_dict(
            {"edges": [{"src": "a.py"}]}, "a.py", self._path_to_ids, self._id_to_change
        )
        assert result == []

    def test_impacted_files_non_strings_skipped(self):
        result = _parse_impact_dict(
            {"impacted_files": [None, 42, "", "b.py"]},
            "a.py",
            self._path_to_ids,
            self._id_to_change,
        )
        assert ("id_a", "id_b") in result
        assert len(result) == 1

    def test_empty_dict_no_pairs(self):
        result = _parse_impact_dict({}, "a.py", self._path_to_ids, self._id_to_change)
        assert result == []

    def test_edges_integers_in_list_skipped(self):
        result = _parse_impact_dict(
            {"edges": [42, "str"]}, "a.py", self._path_to_ids, self._id_to_change
        )
        assert result == []

    def test_source_path_excluded_from_impacted(self):
        result = _parse_impact_dict(
            {"impacted_files": ["a.py", "b.py"]}, "a.py", self._path_to_ids, self._id_to_change
        )
        assert ("id_a", "id_a") not in result
        assert ("id_a", "id_b") in result


# ---------------------------------------------------------------------------
# graph_build.py — build_change_graph edge cases
# ---------------------------------------------------------------------------


class TestBuildChangeGraph:
    def test_crg_raises_propagates(self):
        crg = MagicMock()
        crg.get_impact_radius.side_effect = RuntimeError("CRG exploded")
        a = FileChange(id="a", path="a.py")
        with pytest.raises(RuntimeError, match="CRG exploded"):
            build_change_graph([a], crg=crg)

    def test_crg_empty_dict_no_crash(self):
        crg = MagicMock()
        crg.get_impact_radius.return_value = {}
        a = FileChange(id="a", path="a.py")
        nodes, edges, _ = build_change_graph([a], crg=crg)
        assert edges == ()

    def test_self_loop_new_relationship_filtered(self):
        a = FileChange(id="a", path="a.py")
        rel = NewRelationship(source_change_id="a", dependant_change_id="a", reason="new_call")
        nodes, edges, _ = build_change_graph([a], new_relationships=[rel])
        assert ("a", "a") not in edges
        assert edges == ()

    def test_duplicate_new_relationships_deduplicated(self):
        a = FileChange(id="a", path="a.py")
        b = FileChange(id="b", path="b.py")
        rel = NewRelationship(source_change_id="a", dependant_change_id="b", reason="new_import")
        nodes, edges, _ = build_change_graph([a, b], new_relationships=[rel, rel])
        assert edges.count(("a", "b")) == 1

    def test_unknown_new_relationship_endpoint_raises(self):
        a = FileChange(id="a", path="a.py")
        rel = NewRelationship(source_change_id="a", dependant_change_id="ghost", reason="new_file")
        with pytest.raises(ValueError, match="unknown edge endpoint"):
            build_change_graph([a], new_relationships=[rel])

    def test_unicode_ids_and_paths(self):
        a = FileChange(id="zh_file", path="src/zh.py")
        b = FileChange(id="es_file", path="src/es.py")
        rel = NewRelationship(
            source_change_id="zh_file", dependant_change_id="es_file", reason="new_import"
        )
        nodes, edges, _ = build_change_graph([a, b], new_relationships=[rel])
        assert ("zh_file", "es_file") in edges

    def test_circular_depends_on_creates_both_edges(self):
        a = FileChange(id="a", path="a.py", depends_on=("b",))
        b = FileChange(id="b", path="b.py", depends_on=("a",))
        nodes, edges, _ = build_change_graph([a, b])
        assert ("b", "a") in edges
        assert ("a", "b") in edges


# ---------------------------------------------------------------------------
# contract_sccs — stress and boundary
# ---------------------------------------------------------------------------


class TestContractSCCs:
    def test_large_linear_chain(self):
        N = 1000
        nodes = tuple(str(i) for i in range(N))
        edges = tuple((str(i), str(i + 1)) for i in range(N - 1))
        scc_of, dag_edges = contract_sccs(nodes, edges)
        assert len(set(scc_of.values())) == N
        assert len(dag_edges) == N - 1

    def test_single_node_no_edges(self):
        scc_of, dag_edges = contract_sccs(("x",), ())
        assert scc_of["x"] == "x"
        assert dag_edges == ()

    def test_complete_bidir_graph_one_scc(self):
        nodes = ("a", "b", "c", "d")
        edges = (
            ("a", "b"),
            ("b", "a"),
            ("b", "c"),
            ("c", "b"),
            ("c", "d"),
            ("d", "c"),
        )
        scc_of, dag_edges = contract_sccs(nodes, edges)
        assert len(set(scc_of.values())) == 1
        assert dag_edges == ()

    def test_empty_inputs(self):
        scc_of, dag_edges = contract_sccs((), ())
        assert scc_of == {}
        assert dag_edges == ()


# ---------------------------------------------------------------------------
# solver.py — plan_batches boundary conditions
# ---------------------------------------------------------------------------


class TestPlanBatchesBoundary:
    def test_budget_zero_oversized(self, tmp_path):
        # budget=0: every change exceeds budget, so all become oversized batches
        c = FileChange(id="1", path="a.py", edit_kind="delete")  # 80 tokens
        plan = plan_batches([c], budget=0, root=tmp_path)
        assert plan.solver_status == "OPTIMAL"
        assert len(plan.batches) == 1
        assert plan.batches[0].oversized is True

    def test_budget_negative_oversized(self, tmp_path):
        # negative budget: every change exceeds budget, so all become oversized batches
        c = FileChange(id="1", path="a.py", edit_kind="delete")  # 80 tokens
        plan = plan_batches([c], budget=-1, root=tmp_path)
        assert plan.solver_status == "OPTIMAL"
        assert len(plan.batches) == 1
        assert plan.batches[0].oversized is True

    def test_time_limit_zero_valid_status(self, tmp_path):
        c = FileChange(id="1", path="a.py", edit_kind="delete")
        plan = plan_batches([c], budget=70_000, time_limit_s=0, root=tmp_path)
        assert plan.solver_status in ("OPTIMAL", "FEASIBLE", "UNKNOWN", "INFEASIBLE")

    def test_time_limit_negative_rejected(self, tmp_path):
        c = FileChange(id="1", path="a.py", edit_kind="delete")
        with pytest.raises(ValueError, match="time_limit_s"):
            plan_batches([c], budget=70_000, time_limit_s=-1, root=tmp_path)

    def test_duplicate_paths_different_ids_ok(self, tmp_path):
        a = FileChange(id="a", path="same.py", edit_kind="delete")
        b = FileChange(id="b", path="same.py", edit_kind="delete")
        plan = plan_batches([a, b], budget=70_000, root=tmp_path)
        assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
        all_ids = {cid for batch in plan.batches for cid in batch.change_ids}
        assert all_ids == {"a", "b"}

    def test_circular_depends_on_co_batched(self, tmp_path):
        a = FileChange(id="a", path="a.py", edit_kind="delete", depends_on=("b",))
        b = FileChange(id="b", path="b.py", edit_kind="delete", depends_on=("a",))
        plan = plan_batches([a, b], budget=70_000, root=tmp_path)
        assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
        idx = {cid: batch.index for batch in plan.batches for cid in batch.change_ids}
        assert idx["a"] == idx["b"], "Circular deps must co-batch via SCC"

    def test_budget_exact_fit(self, tmp_path):
        c = FileChange(id="1", path="a.py", edit_kind="delete")  # exactly 80 tokens
        plan = plan_batches([c], budget=80, root=tmp_path)
        assert plan.solver_status in ("OPTIMAL", "FEASIBLE")

    def test_budget_one_under_oversized(self, tmp_path):
        # 80-token change with budget=79: oversized passthrough, not INFEASIBLE
        c = FileChange(id="1", path="a.py", edit_kind="delete")  # exactly 80 tokens
        plan = plan_batches([c], budget=79, root=tmp_path)
        assert plan.solver_status == "OPTIMAL"
        assert len(plan.batches) == 1
        assert plan.batches[0].oversized is True

    def test_split_symbol_spread_reported(self, tmp_path):
        sym = SymbolRef(name="Widget", file="widget.py")
        a = FileChange(id="a", path="a.py", edit_kind="delete", symbols=(sym,))
        b = FileChange(id="b", path="b.py", edit_kind="delete", symbols=(sym,))
        # 80+80=160 > budget=100 -> must split
        plan = plan_batches([a, b], budget=100, root=tmp_path)
        assert plan.solver_status in ("OPTIMAL", "FEASIBLE")
        widget_entry = next(
            ss
            for ss in plan.spread_report
            if ss.symbol.file == "widget.py" and ss.symbol.name == "Widget"
        )
        assert widget_entry.spread == 1

    def test_200_changes_no_hang(self, tmp_path):
        changes = [FileChange(id=str(i), path=f"f{i}.py", edit_kind="delete") for i in range(200)]
        # 200 * 80 = 16000 -> all fit in one batch
        plan = plan_batches(changes, budget=16_000, time_limit_s=5.0, root=tmp_path)
        assert plan.solver_status in ("OPTIMAL", "FEASIBLE", "UNKNOWN")


# ---------------------------------------------------------------------------
# mcp_server.py — _plan_batches_impl input validation
# ---------------------------------------------------------------------------


class TestPlanBatchesImpl:
    @pytest.fixture(autouse=True)
    def _stub_crg(self, monkeypatch):
        from milknado.adapters import crg as crg_mod

        class StubAdapter:
            def __init__(self, project_root) -> None:
                pass

            def get_impact_radius(self, files):
                return {"impacted_files": []}

            def ensure_graph(self, project_root) -> None:
                pass

            def get_architecture_overview(self):
                return {}

        monkeypatch.setattr(crg_mod, "CrgAdapter", StubAdapter)

    def test_missing_id_raises_key_error(self, tmp_path):
        from milknado.mcp_server import _plan_batches_impl

        with pytest.raises(KeyError):
            _plan_batches_impl([{"path": "a.py"}], 70_000, tmp_path)

    def test_missing_path_raises_key_error(self, tmp_path):
        from milknado.mcp_server import _plan_batches_impl

        with pytest.raises(KeyError):
            _plan_batches_impl([{"id": "1"}], 70_000, tmp_path)

    def test_extra_keys_ignored(self, tmp_path):
        from milknado.mcp_server import _plan_batches_impl

        result = _plan_batches_impl(
            [{"id": "1", "path": "a.py", "edit_kind": "delete", "bogus": "trash"}],
            70_000,
            tmp_path,
        )
        assert result["solver_status"] in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN")

    def test_symbol_missing_name_raises(self, tmp_path):
        from milknado.mcp_server import _plan_batches_impl

        with pytest.raises(ValueError, match="must have string"):
            _plan_batches_impl(
                [{"id": "1", "path": "a.py", "symbols": [{"file": "a.py"}]}],
                70_000,
                tmp_path,
            )

    def test_symbol_missing_file_raises(self, tmp_path):
        from milknado.mcp_server import _plan_batches_impl

        with pytest.raises(ValueError, match="must have string"):
            _plan_batches_impl(
                [{"id": "1", "path": "a.py", "symbols": [{"name": "Foo"}]}],
                70_000,
                tmp_path,
            )

    def test_nonexistent_project_root_no_crash(self, tmp_path):
        from milknado.mcp_server import _plan_batches_impl

        nonexistent = tmp_path / "ghost_dir"
        result = _plan_batches_impl(
            [{"id": "1", "path": "a.py", "edit_kind": "delete"}],
            70_000,
            nonexistent,
        )
        assert result["solver_status"] in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN")

    def test_empty_changes_returns_optimal(self, tmp_path):
        from milknado.mcp_server import _plan_batches_impl

        result = _plan_batches_impl([], 70_000, tmp_path)
        assert result["solver_status"] == "OPTIMAL"
        assert result["batches"] == []
