"""Worker follow-up tool with durable source provenance."""

from __future__ import annotations

from dataclasses import dataclass

from fastmcp import Context

from milknado.domains.common import MilknadoConfig
from milknado.domains.graph import FollowUpRequest, FollowUpSource, MikadoGraph
from milknado.mcp._core import Flavor, Kind, Response, mcp, open_graph, resolve_project_root
from milknado.mcp._todo_creation import TodoRequest, create_todo, node_inputs
from milknado.mcp.todo import follow_up_source, node_to_summary


@dataclass(frozen=True)
class _FollowUpCommand:
    request: TodoRequest
    parent_id: int | None
    source: FollowUpSource


def _create_worker_follow_up(
    graph: MikadoGraph, cfg: MilknadoConfig, command: _FollowUpCommand
) -> Response:
    spec, files = node_inputs(cfg, command.request)
    node, created = graph.add_follow_up(
        FollowUpRequest(
            command.request.description,
            command.parent_id,
            spec,
            files or (),
            command.source,
        )
    )
    source = command.source
    source_link = {
        "kind": "worker_invocation",
        "node_id": source.node_id,
        "run_id": source.run_id,
        "invocation_id": source.invocation_id,
        "request_id": source.request_id,
    }
    return {
        **node_to_summary(node),
        "created": created,
        "links": {
            "discovered_node": {"kind": "node", "node_id": node.id},
            "source": source_link,
        },
    }


@mcp.tool()
def milknado_track_follow_up(  # noqa: PLR0913 - MCP tool schema
    description: str,
    kind: Kind = "task",
    parent_id: int | None = None,
    project_root: str = "",
    files: list[str] | None = None,
    flavor: Flavor | None = None,
    artifact: str | None = None,
    prereqs: list[int] | None = None,
    ctx: Context | None = None,
) -> Response:
    """Register discovered work and link it to the calling worker invocation."""
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        source = follow_up_source(ctx.request_id if ctx is not None else "")
        request = TodoRequest(
            description=description,
            kind=kind,
            files=files,
            flavor=flavor,
            artifact=artifact,
            prereqs=tuple(prereqs) if prereqs is not None else None,
            root=root,
        )
        if source is None:
            summary = create_todo(graph, cfg, parent_id, request)
            return {key: value for key, value in summary.items()}
        if ctx is None:
            raise ValueError("MCP request context is required for worker follow-ups")
        return _create_worker_follow_up(graph, cfg, _FollowUpCommand(request, parent_id, source))
    finally:
        graph.close()
