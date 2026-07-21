"""Milknado MCP stdio server — graph tools for Cursor, Codex, Gemini CLI, etc."""

from __future__ import annotations

import logging

from milknado.app.plan_batches import _solve_manifest
from milknado.app.plan_batches import plan_batches as _plan_batches_impl
from milknado.domains.batching import DUMB_ZONE_BUDGET
from milknado.mcp._core import (
    Flavor,
    Kind,
    TodoStatus,
    _parse_flavor,
    _parse_kind,
    _parse_todo_status,
    mcp,
    open_graph,
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
) -> dict:
    """Return a bounded page of Mikado nodes filtered in the graph query.

    ``page`` is ``(limit, offset)``. Each node includes id, status, and
    description; no matches returns an empty nodes list.
    """
    want_status = _parse_todo_status(status) if status is not None else None
    want_kind = _parse_kind(kind) if kind is not None else None
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        want_flavor = _parse_flavor(flavor, cfg.flavor_registry) if flavor is not None else None
        return {
            "nodes": graph.get_node_summaries(
                status=want_status,
                kind=want_kind,
                flavor=want_flavor,
                page=page,
            )
        }
    finally:
        graph.close()


@mcp.tool()
def milknado_plan_batches(
    changes: list[dict],
    budget: int = DUMB_ZONE_BUDGET,
    project_root: str = "",
    new_relationships: list[dict] | None = None,
    force_single_batch: bool = False,
) -> dict:
    """Compute token-budgeted, precedence-respecting batches for changes."""
    root = resolve_project_root(project_root or None)
    return _plan_batches_impl(
        changes, budget, root, new_relationships, force_single_batch=force_single_batch
    )


@mcp.tool()
def milknado_plan_apply(
    manifest: dict,
    project_root: str = "",
    parent_id: int | None = None,
    budget: int = DUMB_ZONE_BUDGET,
    force_single_batch: bool = False,
) -> dict:
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
        summary = {"nodes": graph.get_node_summaries()}
    finally:
        graph.close()
    return {"nodes_created": created, "graph_summary": summary}


def main() -> None:
    from milknado.process_logging import configure_stderr_logging

    # the MCP server speaks stdio — logging must never write to stdout, only stderr/file.
    configure_stderr_logging()

    # Importing each tool module registers its @mcp.tool()s on the shared instance.
    from milknado.mcp import (  # noqa: F401
        github,
        node,
        ralph,
        run,
        todo,
        todo_mutate,
        wiki,
    )

    mcp.run()
