"""Milknado MCP todo tools — registered against the shared FastMCP instance."""

from __future__ import annotations

import os

from milknado._mcp_core import Kind, TodoStatus, mcp, open_graph, resolve_project_root
from milknado.domains.common import VALID_TRANSITIONS, MikadoNode, NodeKind, NodeStatus
from milknado.domains.common.errors import InvalidTransition
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


def _build_subtree(
    node: MikadoNode,
    children_map: dict[int, list[MikadoNode]],
    max_depth: int | None = None,
) -> dict:
    payload = _node_to_summary(node)
    if max_depth is not None and max_depth <= 0:
        payload["children"] = []
        return payload
    child_depth = None if max_depth is None else max_depth - 1
    payload["children"] = [
        _build_subtree(c, children_map, child_depth) for c in children_map.get(node.id, [])
    ]
    return payload


def _todo_status_steps(current: NodeStatus, target: NodeStatus) -> list[NodeStatus]:
    """The ordered state-machine steps moving `current` to `target`.

    The facade exposes pending/in_progress/blocked/done, but the underlying
    state machine only allows BLOCKED -> PENDING and forbids PENDING -> DONE
    directly. Step through PENDING/RUNNING as needed so every reachable facade
    status is reachable. Setting a node to its current status yields no steps.
    """
    if current == target:
        return []
    steps: list[NodeStatus] = []
    if current == NodeStatus.BLOCKED and target in (NodeStatus.RUNNING, NodeStatus.DONE):
        steps.append(NodeStatus.PENDING)
        current = NodeStatus.PENDING
    if target == NodeStatus.DONE and current == NodeStatus.PENDING:
        steps.append(NodeStatus.RUNNING)
    steps.append(target)
    return steps


def _validate_todo_status(node: MikadoNode, target: NodeStatus) -> None:
    """Raise InvalidTransition if moving `node` to `target` is illegal, without
    mutating anything. Walks the same step sequence `_apply_todo_status` would
    take against VALID_TRANSITIONS so a bulk caller can pre-flight a whole
    subtree before any write commits."""
    state = node.status
    for step in _todo_status_steps(node.status, target):
        if step not in VALID_TRANSITIONS.get(state, set()):
            raise InvalidTransition(
                node_id=node.id,
                current=state,
                target=step,
                valid_targets=tuple(VALID_TRANSITIONS.get(state, set())),
            )
        state = step


def _apply_todo_status(graph, node: MikadoNode, target: NodeStatus) -> None:
    """Move a node to one of the todo facade's statuses.

    Steps through PENDING/RUNNING as needed (see `_todo_status_steps`). Setting
    a node to its current status is a no-op. Reopening a DONE node is
    intentionally disallowed and still raises InvalidTransition.
    """
    dispatch = {
        NodeStatus.PENDING: graph.mark_pending,
        NodeStatus.RUNNING: graph.mark_running,
        NodeStatus.DONE: graph.mark_done,
        NodeStatus.BLOCKED: graph.mark_blocked,
    }
    for step in _todo_status_steps(node.status, target):
        dispatch[step](node.id)


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
def milknado_todo_add(
    description: str,
    kind: Kind = "task",
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
def milknado_todo_set_status(node_id: int, status: TodoStatus, project_root: str = "") -> dict:
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


def _subtree_post_order(
    children_map: dict[int, list[MikadoNode]], root: MikadoNode
) -> list[MikadoNode]:
    """Nodes in root's subtree, children before parents, each visited once."""
    visited: set[int] = set()
    ordered: list[MikadoNode] = []

    def visit(node: MikadoNode) -> None:
        if node.id in visited:
            return
        visited.add(node.id)
        for child in children_map.get(node.id, []):
            visit(child)
        ordered.append(node)

    visit(root)
    return ordered


@mcp.tool()
def milknado_set_subtree_status(root_id: int, status: TodoStatus, project_root: str = "") -> dict:
    """Set status on every node in root_id's subtree (post-order, root last).

    Children reach the target before their parents, so marking a goal done sees
    its prerequisites already completed. Returns {"updated": <count>} of nodes
    whose status actually changed; re-running on an unchanged subtree is a no-op.
    Reopening a DONE node still raises (the bulk path reuses the same transition
    rules as milknado_todo_set_status).

    The whole subtree is validated before any write, so an illegal transition
    (e.g. reopening a DONE node) raises with the graph left untouched — the bulk
    update is all-or-nothing.
    """
    target = _parse_todo_status(status)
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(root_id)
        if node is None:
            raise ValueError(f"node {root_id} not found")
        children_map = graph.get_children_map()
        ordered = _subtree_post_order(children_map, node)
        for n in ordered:
            _validate_todo_status(n, target)
        updated = 0
        for n in ordered:
            if n.status != target:
                _apply_todo_status(graph, n, target)
                updated += 1
        return {"updated": updated}
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_next(kind: Kind = "task", project_root: str = "") -> dict | None:
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
            **_node_to_summary(node),
            "parent_id": node.parent_id,
            "prerequisite_ids": [c.id for c in graph.get_children(node_id)],
        }
    finally:
        graph.close()


def _worker_node_id() -> int | None:
    """Read the worker's current node id from MILKNADO_NODE_ID, or None if unset."""
    raw = os.environ.get("MILKNADO_NODE_ID", "").strip()
    return int(raw) if raw else None


@mcp.tool()
def milknado_track_follow_up(
    description: str,
    kind: Kind = "task",
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
def milknado_move_node(node_id: int, new_parent_id: int | None, project_root: str = "") -> dict:
    """Re-parent a node under new_parent_id (None moves it to root level).

    Preserves the node id and its children; rejects a move that would make the
    node its own ancestor (cycle). Returns the moved node's summary.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        graph.move_node(node_id, new_parent_id)
        moved = graph.get_node(node_id)
        if moved is None:
            raise ValueError(f"node {node_id} not found")
        return _node_to_summary(moved)
    finally:
        graph.close()


@mcp.tool()
def milknado_edit_node(
    node_id: int,
    description: str | None = None,
    kind: Kind | None = None,
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
