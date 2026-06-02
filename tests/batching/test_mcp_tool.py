from __future__ import annotations

import logging

import pytest

from milknado.mcp_server import _dict_to_file_change, _plan_batches_impl


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
    """_check_mega_batch must fire via the MCP path when 1 batch > threshold."""
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
    # Lower threshold to 2 so 3 changes in a single budget-fitting batch triggers the guard
    monkeypatch.setattr("milknado.domains.planning.planner.MEGA_BATCH_THRESHOLD", 2)
    changes = [{"id": str(i), "path": f"f{i}.py", "edit_kind": "delete"} for i in range(3)]
    with pytest.raises(MegaBatchAborted):
        _plan_batches_impl(changes, 70_000, tmp_path)


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
    monkeypatch.setattr("milknado.domains.planning.planner.MEGA_BATCH_THRESHOLD", 2)
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
