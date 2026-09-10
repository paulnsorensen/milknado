from __future__ import annotations

from typing import cast

from easy_cheese_schemas import CurdPlan, supported_version_for, validate_contract

from milknado.domains.planning import admit_curd_plan
from milknado.mcp._core import Response, mcp, open_graph, resolve_project_root


@mcp.tool()
def milknado_curd_plan_admit(
    plan_json: str,
    project_root: str = "",
    parent_id: int | None = None,
) -> Response:
    """Strictly validate and admit an Easy Cheese CurdPlan into the graph."""
    supported_version = supported_version_for(CurdPlan)
    if supported_version is None:
        raise ValueError("Easy Cheese does not expose a supported CurdPlan version")
    plan = cast(
        CurdPlan,
        validate_contract(
            plan_json,
            CurdPlan,
            supported_version=supported_version,
        ).value,
    )
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        admission = admit_curd_plan(plan, graph, parent_id=parent_id)
        return {
            "goal_id": admission.goal_id,
            "curd_node_ids": dict(admission.curd_node_ids),
        }
    finally:
        graph.close()


__all__ = ["milknado_curd_plan_admit"]
