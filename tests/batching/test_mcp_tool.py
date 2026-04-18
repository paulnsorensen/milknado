from __future__ import annotations

import pytest

from milknado.mcp_server import _plan_batches_impl


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
