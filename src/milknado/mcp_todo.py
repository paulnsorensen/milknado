"""Milknado MCP todo tools — read tools registered against the shared FastMCP instance."""

from __future__ import annotations

import logging
import os

from milknado._mcp_core import (
    Flavor,
    Kind,
    _parse_flavor,
    _parse_kind,
    mcp,
    open_graph,
    resolve_project_root,
)
from milknado.domains.common import MikadoNode
from milknado.domains.common.flavor_profile import resolve_flavor_profile
from milknado.domains.dispatch import render_brief

_logger = logging.getLogger(__name__)


def node_to_summary(node: MikadoNode) -> dict:
    result: dict = {
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
) -> dict:
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
) -> list[dict]:
    """Return the todo tree from root_id, or the forest of all top-level nodes.

    max_depth bounds how far to descend: 0 returns each root node only, 1 adds
    direct children, and so on; None (default) returns the full subtree.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        children_map = graph.get_children_map()
        if root_id is not None:
            node = graph.get_node(root_id)
            if node is None:
                raise ValueError(f"node {root_id} not found")
            return [_build_subtree(node, children_map, max_depth)]
        return [_build_subtree(n, children_map, max_depth) for n in graph.get_roots()]
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_next(
    kind: Kind = "task",
    flavor: Flavor | None = None,
    project_root: str = "",
) -> dict | None:
    """Return the next runnable node (leaf with no incomplete prereqs).

    flavor: if provided, only returns nodes with the matching flavor.
    """
    node_kind = _parse_kind(kind)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        node_flavor = _parse_flavor(flavor, cfg.flavor_registry) if flavor is not None else None
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
def milknado_todo_brief(node_id: int, project_root: str = "") -> dict:
    """Render a markdown brief for a task (description, ancestor goals, prereqs, files)."""
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        node_for_brief = graph.get_node(node_id)
        flavor = node_for_brief.flavor if node_for_brief is not None else None
        profile = resolve_flavor_profile(cfg, flavor)
        brief = render_brief(graph, node_id, prepend=profile.brief_prepend)
        files = graph.get_file_ownership(node_id)
        return {"node_id": node_id, "brief": brief, "files": files}
    finally:
        graph.close()


@mcp.tool()
def milknado_get_node(node_id: int, project_root: str = "") -> dict:
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


def _worker_node_id() -> int | None:
    """Read the worker's current node id from MILKNADO_NODE_ID, or None if unset."""
    raw = os.environ.get("MILKNADO_NODE_ID", "").strip()
    return int(raw) if raw else None


def follow_up_parent_id(graph) -> int | None:  # noqa: ANN001
    """Parent for an auto-parented follow-up: the worker node's own parent.

    The worker's exit-0 reconcile drives MILKNADO_NODE_ID's node to done, and
    children are prerequisites — so parenting the follow-up under the worker
    node would leave a done node holding unmet work (#124). Attaching it to
    the node's parent keeps it a sibling: it still gates the same goal but
    never gates the completing node. Fails loud on a stale MILKNADO_NODE_ID
    before anything is inserted.
    """
    worker_id = _worker_node_id()
    if worker_id is None:
        return None
    node = graph.get_node(worker_id)
    if node is None:
        raise ValueError(f"MILKNADO_NODE_ID {worker_id} not found")
    return node.parent_id
