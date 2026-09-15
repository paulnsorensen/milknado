"""Milknado MCP todo tools — read tools registered against the shared FastMCP instance."""

from __future__ import annotations

import logging
import os

from milknado.domains.common import MikadoNode, resolve_flavor_profile
from milknado.domains.dispatch import render_brief
from milknado.domains.graph import FollowUpSource
from milknado.mcp._core import (
    Flavor,
    Kind,
    NodeSummary,
    Response,
    mcp,
    open_graph,
    parse_flavor,
    parse_kind,
    resolve_project_root,
)

_logger = logging.getLogger(__name__)


def node_to_summary(node: MikadoNode) -> NodeSummary:
    result: NodeSummary = {
        "id": node.id,
        "kind": node.kind.value,
        "status": node.status.value,
        "description": node.description,
    }
    if node.flavor is not None:
        result["flavor"] = node.flavor
    return result


def _build_subtree(
    node: MikadoNode,
    children_map: dict[int, list[MikadoNode]],
    max_depth: int | None = None,
) -> NodeSummary:
    payload = node_to_summary(node)
    if max_depth is not None and max_depth <= 0:
        payload["children"] = []
        return payload
    child_depth = None if max_depth is None else max_depth - 1
    payload["children"] = [
        _build_subtree(c, children_map, child_depth) for c in children_map.get(node.id, [])
    ]
    return payload


@mcp.tool()
def milknado_todo_tree(
    project_root: str = "",
    root_id: int | None = None,
    max_depth: int | None = None,
    include_archived: bool = False,
) -> list[NodeSummary]:
    """Return the todo tree from root_id, or the forest of all top-level nodes.

    max_depth bounds how far to descend: 0 returns each root node only, 1 adds
    direct children, and so on; None (default) returns the full subtree.
    include_archived surfaces soft-hidden (archived) nodes; default hides them.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        children_map = graph.get_children_map(include_archived=include_archived)
        if root_id is not None:
            node = graph.get_node(root_id)
            if node is None:
                raise ValueError(f"node {root_id} not found")
            if node.archived_at is not None and not include_archived:
                return []
            return [_build_subtree(node, children_map, max_depth)]
        return [
            _build_subtree(n, children_map, max_depth)
            for n in graph.get_roots(include_archived=include_archived)
        ]
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_next(
    kind: Kind = "task",
    flavor: Flavor | None = None,
    project_root: str = "",
) -> NodeSummary | None:
    """Return the next runnable node (leaf with no incomplete prereqs).

    flavor: if provided, only returns nodes with the matching flavor.
    """
    node_kind = parse_kind(kind)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        node_flavor = parse_flavor(flavor, cfg.flavor_registry) if flavor is not None else None
        for node in graph.get_ready_nodes():
            if node.kind != node_kind:
                continue
            if node_flavor is not None and node.flavor != node_flavor:
                continue
            return node_to_summary(node)
        return None
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_brief(node_id: int, project_root: str = "") -> Response:
    """Render a markdown brief for a task (description, ancestor goals, prereqs, files)."""
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        node_for_brief = graph.get_node(node_id)
        flavor = node_for_brief.flavor if node_for_brief is not None else None
        profile = resolve_flavor_profile(cfg, flavor)
        brief = render_brief(
            graph,
            node_id,
            prepend=profile.brief_prepend,
            project_root=root,
        )
        files = graph.files.for_node(node_id)
        return {"node_id": node_id, "brief": brief, "files": files}
    finally:
        graph.close()


@mcp.tool()
def milknado_get_node(node_id: int, project_root: str = "") -> Response:
    """Read one node by id: summary fields plus parent_id and prerequisite_ids.

    A node's prerequisites are its children (a node is ready once all children
    are done), so prerequisite_ids is the list of child ids.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        return {
            **node_to_summary(node),
            "parent_id": node.parent_id,
            "prerequisite_ids": [c.id for c in graph.get_children(node_id)],
        }
    finally:
        graph.close()


def follow_up_source(request_id: str | int) -> FollowUpSource | None:
    """Read one complete worker identity from the inherited process context."""
    names = (
        "MILKNADO_NODE_ID",
        "MILKNADO_RUN_ID",
        "MILKNADO_INVOCATION_ID",
    )
    node_id, run_id, invocation_id = (os.environ.get(name, "").strip() for name in names)
    if not any((node_id, run_id, invocation_id)):
        return None
    missing = [
        name
        for name, value in zip(names, (node_id, run_id, invocation_id), strict=True)
        if not value
    ]
    if missing:
        raise ValueError(f"incomplete follow-up worker context: missing {', '.join(missing)}")
    try:
        parsed_node_id = int(node_id)
    except ValueError as exc:
        raise ValueError(f"invalid MILKNADO_NODE_ID {node_id!r}") from exc
    return FollowUpSource(parsed_node_id, run_id, invocation_id, str(request_id))
