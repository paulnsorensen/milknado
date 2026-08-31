"""Milknado MCP stdio server — graph tools for Cursor, Codex, Gemini CLI, etc."""

from __future__ import annotations

import logging
from typing import cast

from milknado.app.plan_batches import _solve_manifest  # pyright: ignore[reportPrivateUsage]
from milknado.app.plan_batches import (
    plan_batches as _plan_batches_impl,  # pyright: ignore[reportUnknownVariableType]
)
from milknado.domains.batching import DUMB_ZONE_BUDGET
from milknado.mcp._core import (
    BatchPlanResponse,
    Flavor,
    GraphNodeSummary,
    GraphSummaryResponse,
    Kind,
    PlanApplyResponse,
    TodoStatus,
    mcp,
    open_graph,
    parse_flavor,
    parse_kind,
    parse_todo_status,
    resolve_project_root,
)

__all__ = ["main", "mcp", "open_graph", "resolve_project_root"]

_logger = logging.getLogger(__name__)


@mcp.tool()
def milknado_graph_summary(
    project_root: str = "",
    status: TodoStatus | None = None,
    kind: Kind | None = None,
    flavor: Flavor | None = None,
    page: tuple[int, int] = (100, 0),
    include_archived: bool = False,
) -> GraphSummaryResponse:
    """Return a bounded page of Mikado nodes filtered in the graph query.

    ``page`` is ``(limit, offset)``. Each node includes id, status, and
    description; no matches returns an empty nodes list. ``include_archived``
    surfaces soft-hidden (archived) nodes; default hides them.
    """
    want_status = parse_todo_status(status) if status is not None else None
    want_kind = parse_kind(kind) if kind is not None else None
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        want_flavor = parse_flavor(flavor, cfg.flavor_registry) if flavor is not None else None
        return {
            "nodes": cast(
                list[GraphNodeSummary],
                cast(
                    object,
                    graph.get_node_summaries(
                        status=want_status,
                        kind=want_kind,
                        flavor=want_flavor,
                        page=page,
                        include_archived=include_archived,
                    ),
                ),
            )
        }
    finally:
        graph.close()


@mcp.tool()
def milknado_plan_batches(
    changes: list[dict[str, object]],
    budget: int = DUMB_ZONE_BUDGET,
    project_root: str = "",
    new_relationships: list[dict[str, object]] | None = None,
    force_single_batch: bool = False,
) -> BatchPlanResponse:
    """Compute token-budgeted, precedence-respecting batches for changes."""
    root = resolve_project_root(project_root or None)
    return cast(
        BatchPlanResponse,
        cast(
            object,
            _plan_batches_impl(
                changes, budget, root, new_relationships, force_single_batch=force_single_batch
            ),
        ),
    )


@mcp.tool()
def milknado_plan_apply(
    manifest: dict[str, object],
    project_root: str = "",
    parent_id: int | None = None,
    budget: int = DUMB_ZONE_BUDGET,
    force_single_batch: bool = False,
) -> PlanApplyResponse:
    """Apply a caller-produced milknado.plan.v2 manifest to the graph as Mikado nodes.

    Solves the manifest into precedence-respecting batches and writes them as nodes:
    parent_id=None creates a GOAL root from goal_summary with TASK children; a valid
    parent_id attaches TASK children under it. Returns {nodes_created, graph_summary}.
    """
    from milknado.domains.planning import apply_batches_to_graph, decode_manifest

    root = resolve_project_root(project_root or None)
    parsed = decode_manifest(manifest)
    plan = _solve_manifest(parsed, budget, root, force_single_batch=force_single_batch)
    graph, _cfg = open_graph(root)
    try:
        created = apply_batches_to_graph(graph, plan, parsed, parent_id=parent_id)
        summary: GraphSummaryResponse = {
            "nodes": cast(list[GraphNodeSummary], cast(object, graph.get_node_summaries()))
        }
    finally:
        graph.close()
    return {"nodes_created": created, "graph_summary": summary}


def main() -> None:
    from milknado.process_logging import configure_stderr_logging

    # the MCP server speaks stdio — logging must never write to stdout, only stderr/file.
    _ = configure_stderr_logging()

    # Importing each tool module registers its @mcp.tool()s on the shared instance.
    from milknado.mcp import (  # noqa: V104
        github,
        node,
        ralph,
        rebalance,
        run,
        todo,
        todo_mutate,
        wiki,
    )

    _ = (github, node, ralph, rebalance, run, todo, todo_mutate, wiki)

    _ = mcp.run()
