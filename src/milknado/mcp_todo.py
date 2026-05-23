"""Milknado MCP todo tools — registered against the shared FastMCP instance."""

from __future__ import annotations

import os

from milknado._mcp_core import mcp, open_graph, resolve_project_root
from milknado.domains.common import MikadoNode, NodeKind, NodeStatus
from milknado.domains.dispatch import render_brief

_TODO_STATUS_MAP = {
    "pending": NodeStatus.PENDING,
    "in_progress": NodeStatus.RUNNING,
    "blocked": NodeStatus.BLOCKED,
    "done": NodeStatus.DONE,
}


def _parse_kind(value: str) -> NodeKind:
    try:
        return NodeKind(value)
    except ValueError as exc:
        valid = sorted(k.value for k in NodeKind)
        raise ValueError(f"invalid kind {value!r}; expected one of {valid}") from exc


def _parse_todo_status(value: str) -> NodeStatus:
    if value not in _TODO_STATUS_MAP:
        raise ValueError(f"invalid status {value!r}; expected one of {sorted(_TODO_STATUS_MAP)}")
    return _TODO_STATUS_MAP[value]


def _node_to_summary(node: MikadoNode) -> dict:
    return {
        "id": node.id,
        "kind": node.kind.value,
        "status": node.status.value,
        "description": node.description,
    }


def _build_subtree(node: MikadoNode, children_map: dict[int, list[MikadoNode]]) -> dict:
    payload = _node_to_summary(node)
    payload["children"] = [_build_subtree(c, children_map) for c in children_map.get(node.id, [])]
    return payload


def _apply_todo_status(graph, node: MikadoNode, target: NodeStatus) -> None:
    """Move a node to one of the todo facade's statuses.

    The facade exposes pending/in_progress/blocked/done, but the underlying
    state machine only allows BLOCKED -> PENDING and forbids PENDING -> DONE
    directly. Step through PENDING/RUNNING as needed so every reachable facade
    status is reachable. Setting a node to its current status is a no-op.
    Reopening a DONE node is intentionally disallowed and still raises
    InvalidTransition.
    """
    current = node.status
    if current == target:
        return
    if current == NodeStatus.BLOCKED and target in (NodeStatus.RUNNING, NodeStatus.DONE):
        graph.mark_pending(node.id)
        current = NodeStatus.PENDING
    if target == NodeStatus.DONE and current == NodeStatus.PENDING:
        graph.mark_running(node.id)
    dispatch = {
        NodeStatus.PENDING: graph.mark_pending,
        NodeStatus.RUNNING: graph.mark_running,
        NodeStatus.DONE: graph.mark_done,
        NodeStatus.BLOCKED: graph.mark_blocked,
    }
    dispatch[target](node.id)


@mcp.tool()
def milknado_todo_tree(project_root: str = "", root_id: int | None = None) -> list[dict]:
    """Return the todo tree from root_id, or the forest of all top-level nodes."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        children_map = graph.get_children_map()
        if root_id is not None:
            node = graph.get_node(root_id)
            if node is None:
                raise ValueError(f"node {root_id} not found")
            return [_build_subtree(node, children_map)]
        return [_build_subtree(n, children_map) for n in graph.get_roots()]
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_add(
    description: str,
    kind: str = "task",
    parent_id: int | None = None,
    project_root: str = "",
) -> dict:
    """Add a todo node (kind: roadmap|goal|task), optionally linked under parent_id."""
    node_kind = _parse_kind(kind)
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.add_node(description, parent_id=parent_id, kind=node_kind)
        return _node_to_summary(node)
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_set_status(node_id: int, status: str, project_root: str = "") -> dict:
    """Set todo status: pending | in_progress | blocked | done."""
    target = _parse_todo_status(status)
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        _apply_todo_status(graph, node, target)
        updated = graph.get_node(node_id)
        assert updated is not None
        return _node_to_summary(updated)
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_next(kind: str = "task", project_root: str = "") -> dict | None:
    """Return the next runnable node (leaf with no incomplete prereqs)."""
    node_kind = _parse_kind(kind)
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_next_runnable(node_kind)
        return _node_to_summary(node) if node else None
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_brief(node_id: int, project_root: str = "") -> dict:
    """Render a markdown brief for a task (description, ancestor goals, prereqs, files)."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        brief = render_brief(graph, node_id)
        files = graph.get_file_ownership(node_id)
        return {"node_id": node_id, "brief": brief, "files": files}
    finally:
        graph.close()


def _worker_node_id() -> int | None:
    """Read the worker's current node id from MILKNADO_NODE_ID, or None if unset."""
    raw = os.environ.get("MILKNADO_NODE_ID", "").strip()
    return int(raw) if raw else None


@mcp.tool()
def milknado_track_follow_up(
    description: str,
    kind: str = "task",
    parent_id: int | None = None,
    project_root: str = "",
) -> dict:
    """Register discovered follow-up work as a new node.

    When parent_id is omitted, default it to the worker's current node from
    MILKNADO_NODE_ID; if that is unset the node is created at root level.
    """
    node_kind = _parse_kind(kind)
    if parent_id is None:
        parent_id = _worker_node_id()
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.add_node(description, parent_id=parent_id, kind=node_kind)
        return _node_to_summary(node)
    finally:
        graph.close()


@mcp.tool()
def milknado_delete_node(node_id: int, cascade: bool = False, project_root: str = "") -> dict:
    """Delete a node (coordinator-only). Refuses a node with children unless cascade=True."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        return {"deleted": graph.delete_node(node_id, cascade=cascade)}
    finally:
        graph.close()


@mcp.tool()
def milknado_edit_node(
    node_id: int,
    description: str | None = None,
    kind: str | None = None,
    project_root: str = "",
) -> dict:
    """Edit a node's description and/or kind (coordinator-only). Status is not editable."""
    if description is None and kind is None:
        raise ValueError("nothing to edit: provide description and/or kind")
    node_kind = _parse_kind(kind) if kind is not None else None
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        graph.update_node(node_id, description=description, kind=node_kind)
        updated = graph.get_node(node_id)
        assert updated is not None
        return _node_to_summary(updated)
    finally:
        graph.close()
