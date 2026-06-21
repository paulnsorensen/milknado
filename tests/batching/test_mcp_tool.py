from __future__ import annotations

import logging

import pytest

from milknado.mcp_server import (
    _dict_to_file_change,
    _plan_batches_impl,
    milknado_plan_apply,
)


def _stub_crg(monkeypatch) -> None:
    from milknado.adapters import crg as crg_mod

    class _StubCrg:
        def __init__(self, r) -> None:
            pass

        def ensure_graph(self, r) -> None:
            pass

        def get_impact_radius(self, files):
            return {}

    monkeypatch.setattr(crg_mod, "CrgAdapter", _StubCrg)


def _apply(**kwargs):
    """Call the milknado_plan_apply tool via its underlying Python callable."""
    fn = getattr(milknado_plan_apply, "fn", milknado_plan_apply)
    return fn(**kwargs)


def _valid_manifest() -> dict:
    return {
        "manifest_version": "milknado.plan.v2",
        "goal": "Refactor foo",
        "goal_summary": "Move foo into its own module",
        "changes": [
            {"id": "c1", "path": "src/a.py", "description": "Add module A"},
            {"id": "c2", "path": "src/b.py", "description": "Add module B", "depends_on": ["c1"]},
        ],
    }


def test_plan_batches_impl_stub(tmp_path, monkeypatch) -> None:
    """Test _plan_batches_impl with a stubbed CrgAdapter."""
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

    result = _plan_batches_impl(
        [{"id": "1", "path": "a.py", "edit_kind": "delete"}],
        70_000,
        tmp_path,
    )
    assert result["solver_status"] in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN")
    assert "batches" in result
    assert "spread_report" in result
    # Verify new batch shape
    if result["batches"]:
        first = result["batches"][0]
        assert "index" in first
        assert "change_ids" in first
        assert "depends_on" in first
        assert "oversized" in first
    # spread_report is a list (not dict) in new shape
    assert isinstance(result["spread_report"], list)


@pytest.mark.asyncio
async def test_tool_via_fastmcp_client(tmp_path, monkeypatch) -> None:
    """Test milknado_plan_batches tool end-to-end via FastMCP Client."""
    from milknado.adapters import crg as crg_mod
    from milknado.mcp_server import mcp

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
    # Ensure the tool resolves to tmp_path for file ops
    monkeypatch.setenv("MILKNADO_PROJECT_ROOT", str(tmp_path))

    import json

    from fastmcp import Client

    async with Client(mcp) as c:
        result = await c.call_tool(
            "milknado_plan_batches",
            {"changes": [{"id": "1", "path": "a.py", "edit_kind": "delete"}], "budget": 70_000},
        )

    assert result is not None
    # Extract data from CallToolResult
    content = getattr(result, "content", None) or []
    raw = content[0].text if content and hasattr(content[0], "text") else None
    data = json.loads(raw) if isinstance(raw, str) else {}
    assert data.get("solver_status") in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN")
    assert "batches" in data
    assert isinstance(data.get("spread_report"), list)


# ---------------------------------------------------------------------------
# #66 — MCP-path field round-trip (hash_anchors, dependencies, description)
# ---------------------------------------------------------------------------


def test_hash_anchors_round_trip() -> None:
    """hash_anchors survive the dict→FileChange boundary."""
    fc = _dict_to_file_change(
        {
            "id": "c1",
            "path": "src/foo.py",
            "edit_kind": "modify",
            "hash_anchors": {"before": "sha256:aaa", "after": "sha256:bbb"},
        }
    )
    assert fc.hash_anchors is not None
    assert fc.hash_anchors.before == "sha256:aaa"
    assert fc.hash_anchors.after == "sha256:bbb"


def test_description_round_trip() -> None:
    """description survives the dict→FileChange boundary."""
    fc = _dict_to_file_change({"id": "c1", "path": "src/foo.py", "description": "fix the widget"})
    assert fc.description == "fix the widget"


def test_dependencies_round_trip() -> None:
    """dependencies survive the dict→FileChange boundary with all sub-fields."""
    fc = _dict_to_file_change(
        {
            "id": "c1",
            "path": "src/foo.py",
            "dependencies": [
                {
                    "path": "src/bar.py",
                    "symbols": [{"name": "bar_fn", "file": "src/bar.py"}],
                    "hash_anchors": {"before": "sha256:111", "after": "sha256:222"},
                    "reason": "constrains foo",
                }
            ],
        }
    )
    assert len(fc.dependencies) == 1
    dep = fc.dependencies[0]
    assert dep.path == "src/bar.py"
    assert len(dep.symbols) == 1
    assert dep.symbols[0].name == "bar_fn"
    assert dep.hash_anchors is not None
    assert dep.hash_anchors.before == "sha256:111"
    assert dep.reason == "constrains foo"


def test_all_file_change_fields_populated() -> None:
    """Every non-default FileChange field carries through the MCP boundary."""
    fc = _dict_to_file_change(
        {
            "id": "c1",
            "path": "src/foo.py",
            "edit_kind": "modify",
            "description": "why this change",
            "symbols": [{"name": "Foo", "file": "src/foo.py"}],
            "hash_anchors": {"before": "sha256:aaa", "after": "sha256:bbb"},
            "dependencies": [{"path": "src/dep.py", "reason": "call site"}],
            "depends_on": [],
        }
    )
    assert fc.id == "c1"
    assert fc.path == "src/foo.py"
    assert fc.edit_kind == "modify"
    assert fc.description == "why this change"
    assert fc.symbols[0].name == "Foo"
    assert fc.hash_anchors is not None
    assert len(fc.dependencies) == 1


# ---------------------------------------------------------------------------
# #74 — clear ValueError on missing id/path
# ---------------------------------------------------------------------------


def test_missing_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="id"):
        _dict_to_file_change({"path": "a.py"})


def test_missing_path_raises_value_error() -> None:
    with pytest.raises(ValueError, match="path"):
        _dict_to_file_change({"id": "1"})


def test_empty_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="id"):
        _dict_to_file_change({"id": "", "path": "a.py"})


# ---------------------------------------------------------------------------
# #76 — traversal guard on symbols[].file
# ---------------------------------------------------------------------------


def test_symbol_file_traversal_rejected() -> None:
    with pytest.raises(ValueError, match="traversal"):
        _dict_to_file_change(
            {
                "id": "c1",
                "path": "src/foo.py",
                "symbols": [{"name": "Foo", "file": "../../etc/passwd"}],
            }
        )


def test_symbol_file_absolute_rejected() -> None:
    with pytest.raises(ValueError, match="traversal"):
        _dict_to_file_change(
            {
                "id": "c1",
                "path": "src/foo.py",
                "symbols": [{"name": "Foo", "file": "/etc/passwd"}],
            }
        )


# ---------------------------------------------------------------------------
# #68 — MCP path parity: mega-batch guard fires; telemetry is written
# ---------------------------------------------------------------------------


def test_mega_batch_guard_fires(tmp_path, monkeypatch) -> None:
    """MCP aborts via the BatchPlan property when a batch exceeds the threshold.

    Also pins that the reaction passes the detected change count into the error,
    not a stale or hardcoded value.
    """
    from milknado.adapters import crg as crg_mod
    from milknado.domains.common.errors import MegaBatchAborted

    class _StubCrg:
        def __init__(self, r):
            pass

        def ensure_graph(self, r):
            pass

        def get_impact_radius(self, files):
            return {}

    monkeypatch.setattr(crg_mod, "CrgAdapter", _StubCrg)
    # Lower threshold to 2 so 3 changes in a single budget-fitting batch triggers the guard.
    # The property reads change.MEGA_BATCH_THRESHOLD, so that is the binding to patch.
    monkeypatch.setattr("milknado.domains.batching.change.MEGA_BATCH_THRESHOLD", 2)
    changes = [{"id": str(i), "path": f"f{i}.py", "edit_kind": "delete"} for i in range(3)]
    with pytest.raises(MegaBatchAborted) as exc_info:
        _plan_batches_impl(changes, 70_000, tmp_path)
    assert exc_info.value.change_count == 3


def test_mega_batch_guard_bypassed_by_force(tmp_path, monkeypatch) -> None:
    """force_single_batch=True bypasses the mega-batch guard."""
    from milknado.adapters import crg as crg_mod

    class _StubCrg:
        def __init__(self, r):
            pass

        def ensure_graph(self, r):
            pass

        def get_impact_radius(self, files):
            return {}

    monkeypatch.setattr(crg_mod, "CrgAdapter", _StubCrg)
    monkeypatch.setattr("milknado.domains.batching.change.MEGA_BATCH_THRESHOLD", 2)
    changes = [{"id": str(i), "path": f"f{i}.py", "edit_kind": "delete"} for i in range(3)]
    result = _plan_batches_impl(changes, 70_000, tmp_path, force_single_batch=True)
    assert result["solver_status"] in ("OPTIMAL", "FEASIBLE")


def test_telemetry_written_via_mcp(tmp_path, monkeypatch) -> None:
    """record_batch_snapshot must be called; calibration.jsonl appears on disk."""
    from milknado.adapters import crg as crg_mod

    class _StubCrg:
        def __init__(self, r):
            pass

        def ensure_graph(self, r):
            pass

        def get_impact_radius(self, files):
            return {}

    monkeypatch.setattr(crg_mod, "CrgAdapter", _StubCrg)
    _plan_batches_impl([{"id": "1", "path": "a.py", "edit_kind": "delete"}], 70_000, tmp_path)
    cal = tmp_path / ".milknado" / "calibration.jsonl"
    assert cal.exists(), "calibration.jsonl must be written by the MCP path"


# ---------------------------------------------------------------------------
# #71 / #79 — CRG failure surfaced (not swallowed silently)
# ---------------------------------------------------------------------------


def test_crg_failure_logged_not_swallowed(tmp_path, monkeypatch, caplog) -> None:
    """When CRG raises (including with stderr), a WARNING is emitted and planning continues."""
    from milknado.adapters import crg as crg_mod

    class _BrokenCrg:
        def __init__(self, r):
            pass

        def ensure_graph(self, r) -> None:
            raise RuntimeError("code-review-graph 'build' failed (exit 1): graph error details")

        def get_impact_radius(self, files):
            return {}

    monkeypatch.setattr(crg_mod, "CrgAdapter", _BrokenCrg)
    with caplog.at_level(logging.WARNING, logger="milknado.mcp_server"):
        result = _plan_batches_impl(
            [{"id": "1", "path": "a.py", "edit_kind": "delete"}], 70_000, tmp_path
        )
    assert result["solver_status"] in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN")
    assert any("CRG unavailable" in r.message for r in caplog.records), (
        "A WARNING must be logged when CRG fails"
    )


def test_crg_stderr_in_runtime_error(tmp_path) -> None:
    """_run_crg re-raises CalledProcessError as RuntimeError including stderr."""
    import subprocess
    from unittest.mock import patch

    from milknado.adapters.crg import CrgAdapter

    adapter = CrgAdapter(tmp_path)
    err = subprocess.CalledProcessError(1, ["code-review-graph", "build"], stderr="graph panic!")
    with patch("subprocess.run", side_effect=err):
        with pytest.raises(RuntimeError, match="graph panic!"):
            adapter._run_crg("build")


# ---------------------------------------------------------------------------
# #149 — milknado_plan_apply: solve a manifest dict and land it as graph nodes
# ---------------------------------------------------------------------------


def test_plan_apply_creates_goal_root_and_tasks(tmp_path, monkeypatch) -> None:
    """parent_id=None builds a GOAL root from goal_summary plus TASK children."""
    from milknado.domains.common import NodeKind
    from milknado.mcp_server import open_graph

    _stub_crg(monkeypatch)
    result = _apply(manifest=_valid_manifest(), project_root=str(tmp_path))

    created = result["nodes_created"]
    assert len(created) >= 2  # goal root + at least one task batch
    assert isinstance(result["graph_summary"], dict)
    assert "nodes" in result["graph_summary"]
    assert len(result["graph_summary"]["nodes"]) == len(created)

    graph, _ = open_graph(tmp_path)
    try:
        root_node = graph.get_node(created[0])
        assert root_node is not None
        assert root_node.kind == NodeKind.GOAL
        assert root_node.description == "Move foo into its own module"
        for nid in created[1:]:
            child = graph.get_node(nid)
            assert child is not None
            assert child.kind == NodeKind.TASK
    finally:
        graph.close()


def test_plan_apply_with_parent_id_attaches_tasks_no_goal_root(tmp_path, monkeypatch) -> None:
    """A valid TASK-accepting parent_id attaches TASK children and creates no GOAL root."""
    from milknado.domains.common import NodeKind
    from milknado.mcp_server import open_graph

    _stub_crg(monkeypatch)
    # First apply creates a GOAL root we can attach a second batch of tasks under.
    first = _apply(manifest=_valid_manifest(), project_root=str(tmp_path))
    goal_root_id = first["nodes_created"][0]

    second_manifest = {
        "manifest_version": "milknado.plan.v2",
        "goal": "More work",
        "goal_summary": "Second wave",
        "changes": [{"id": "d1", "path": "src/c.py", "description": "Add module C"}],
    }
    result = _apply(
        manifest=second_manifest,
        project_root=str(tmp_path),
        parent_id=goal_root_id,
    )

    created = result["nodes_created"]
    graph, _ = open_graph(tmp_path)
    try:
        # No new GOAL root — every created node is a TASK.
        for nid in created:
            node = graph.get_node(nid)
            assert node is not None
            assert node.kind == NodeKind.TASK
        # The new task is reachable as a descendant of the existing goal root.
        assert any(c.id in created for c in graph.get_children(goal_root_id))
    finally:
        graph.close()


def test_plan_apply_nonexistent_parent_id_raises(tmp_path, monkeypatch) -> None:
    """A parent_id with no matching node raises ValueError at apply time."""
    _stub_crg(monkeypatch)
    with pytest.raises(ValueError, match="parent"):
        _apply(manifest=_valid_manifest(), project_root=str(tmp_path), parent_id=99999)


def test_plan_apply_kind_invalid_parent_id_raises(tmp_path, monkeypatch) -> None:
    """A parent whose kind rejects TASK children fails loud at the tool boundary.

    ROADMAP accepts only GOAL children, so attaching TASK batches under it must raise.
    This pins the tool's own ValueError path, not only the bridge's.
    """
    from milknado.domains.common import NodeKind
    from milknado.mcp_server import open_graph

    _stub_crg(monkeypatch)
    graph, _ = open_graph(tmp_path)
    try:
        roadmap = graph.add_node("roadmap root", kind=NodeKind.ROADMAP)
        roadmap_id = roadmap.id
    finally:
        graph.close()

    with pytest.raises(ValueError, match="TASK is not a valid child"):
        _apply(manifest=_valid_manifest(), project_root=str(tmp_path), parent_id=roadmap_id)


def test_plan_apply_graph_summary_reflects_created_nodes(tmp_path, monkeypatch) -> None:
    """graph_summary mirrors graph state: GOAL root + node fields round-trip faithfully."""
    from milknado.mcp_server import open_graph

    _stub_crg(monkeypatch)
    result = _apply(manifest=_valid_manifest(), project_root=str(tmp_path))

    created = set(result["nodes_created"])
    summary_nodes = result["graph_summary"]["nodes"]
    assert {n["id"] for n in summary_nodes} == created

    # Each summary entry matches the persisted node exactly (id, status, description).
    graph, _ = open_graph(tmp_path)
    try:
        for entry in summary_nodes:
            node = graph.get_node(entry["id"])
            assert node is not None
            assert entry["status"] == node.status.value
            assert entry["description"] == node.description
    finally:
        graph.close()

    # The GOAL root's goal_summary surfaces in the summary, not just nodes_created.
    root_id = result["nodes_created"][0]
    root_entry = next(n for n in summary_nodes if n["id"] == root_id)
    assert root_entry["description"] == "Move foo into its own module"
    assert root_entry["status"] == "pending"


def test_plan_apply_invalid_manifest_raises_value_error(tmp_path, monkeypatch) -> None:
    """An invalid manifest dict fails loud at the tool boundary, not a silent None."""
    _stub_crg(monkeypatch)
    bad = {"manifest_version": "wrong.version", "goal": "x", "goal_summary": "y", "changes": []}
    with pytest.raises(ValueError, match="milknado.plan.v2"):
        _apply(manifest=bad, project_root=str(tmp_path))


def test_plan_apply_mega_batch_aborts(tmp_path, monkeypatch) -> None:
    """A single oversized batch over the threshold raises MegaBatchAborted unless forced."""
    from milknado.domains.common.errors import MegaBatchAborted

    _stub_crg(monkeypatch)
    monkeypatch.setattr("milknado.domains.batching.change.MEGA_BATCH_THRESHOLD", 2)
    manifest = {
        "manifest_version": "milknado.plan.v2",
        "goal": "Big",
        "goal_summary": "Big change set",
        "changes": [
            {"id": str(i), "path": f"f{i}.py", "edit_kind": "delete", "description": f"change {i}"}
            for i in range(3)
        ],
    }
    with pytest.raises(MegaBatchAborted):
        _apply(manifest=manifest, project_root=str(tmp_path), budget=70_000)


def test_plan_apply_force_single_batch_bypasses_mega_guard(tmp_path, monkeypatch) -> None:
    """force_single_batch=True lands the nodes despite exceeding the mega-batch threshold."""
    from milknado.mcp_server import open_graph

    _stub_crg(monkeypatch)
    monkeypatch.setattr("milknado.domains.batching.change.MEGA_BATCH_THRESHOLD", 2)
    manifest = {
        "manifest_version": "milknado.plan.v2",
        "goal": "Big",
        "goal_summary": "Big change set",
        "changes": [
            {"id": str(i), "path": f"f{i}.py", "edit_kind": "delete", "description": f"change {i}"}
            for i in range(3)
        ],
    }
    result = _apply(
        manifest=manifest,
        project_root=str(tmp_path),
        budget=70_000,
        force_single_batch=True,
    )
    created = result["nodes_created"]
    assert len(created) >= 1
    # The bypass must actually persist nodes, not just return ids.
    graph, _ = open_graph(tmp_path)
    try:
        persisted = {n.id for n in graph.get_all_nodes()}
    finally:
        graph.close()
    assert set(created) <= persisted


def test_plan_apply_traversal_path_raises_value_error(tmp_path, monkeypatch) -> None:
    """A manifest with a traversal path in changes[].path is rejected at the tool boundary."""
    _stub_crg(monkeypatch)
    manifest = {
        "manifest_version": "milknado.plan.v2",
        "goal": "Escape",
        "goal_summary": "Try to escape",
        "changes": [
            {"id": "c1", "path": "../escape.py", "description": "Traversal attempt"},
        ],
    }
    with pytest.raises(ValueError):
        _apply(manifest=manifest, project_root=str(tmp_path))
