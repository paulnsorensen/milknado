"""Milknado MCP todo mutate tools — write tools registered against the shared FastMCP instance."""

from __future__ import annotations

import logging
from pathlib import Path

from milknado._mcp_core import (
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
from milknado.domains.common import NodeKind, NodeSpec, NodeStatus
from milknado.domains.common.paths import normalize_hint_paths
from milknado.domains.graph.status_flow import (
    apply_todo_status,
    subtree_post_order,
    validate_todo_status,
)
from milknado.mcp_node import assert_done_verified
from milknado.mcp_todo import follow_up_parent_id, node_to_summary

_logger = logging.getLogger(__name__)


def _reject_escaping_artifact(artifact: str | None, project_root: Path) -> None:
    """Raise ValueError if *artifact* escapes *project_root*; does not transform it.

    Boundary rejection only: artifact_path is persisted verbatim (existing
    contract), never checked for existence, never resolved before storage.
    """
    if artifact is None:
        return
    raw = artifact.strip()
    if not raw:
        raise ValueError("artifact path is empty")
    p = Path(raw)
    resolved = (project_root / p if not p.is_absolute() else p).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError:
        raise ValueError(f"artifact {artifact!r} escapes project root {project_root}") from None


@mcp.tool()
def milknado_todo_add(
    description: str,
    kind: Kind = "task",
    parent_id: int | None = None,
    project_root: str = "",
    files: list[str] | None = None,
    flavor: Flavor | None = None,
    artifact: str | None = None,
    prereqs: list[int] | None = None,
) -> dict:
    """Add a todo node (kind: roadmap|goal|task), optionally linked under parent_id.

    files: optional list of paths this node will touch (relative to project root,
    or absolute under it).
    flavor: task flavor (implement|spec|spike|prototype|research); only valid for kind=task.
    artifact: opaque repo-relative path to a markdown narrative for this node; persisted
    verbatim, not checked for existence.
    prereqs: node ids that must complete before this node is ready; rejected if any
    would create a cycle.
    """
    node_kind = _parse_kind(kind)
    root = resolve_project_root(project_root or None)
    _reject_escaping_artifact(artifact, root)
    graph, cfg = open_graph(root)
    try:
        node_flavor = _parse_flavor(flavor, cfg.flavor_registry) if flavor is not None else None
        node = graph.add_node(
            description,
            parent_id=parent_id,
            spec=NodeSpec(
                kind=node_kind,
                flavor=node_flavor,
                artifact_path=artifact,
                prereqs=prereqs or (),
                flavor_registry=cfg.flavor_registry,
            ),
        )
        if files is not None:
            graph.set_file_ownership(node.id, normalize_hint_paths(files, root))
        return node_to_summary(node)
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
        if target == NodeStatus.DONE:
            assert_done_verified(graph, node)
        apply_todo_status(graph, node, target)
        updated = graph.get_node(node_id)
        if updated is None:
            raise ValueError(f"node {node_id} not found after status update")
        _logger.info(
            "milknado_todo_set_status: node=%d %s->%s", node_id, node.status.value, status
        )
        return node_to_summary(updated)
    finally:
        graph.close()


@mcp.tool()
def milknado_set_subtree_status(root_id: int, status: TodoStatus, project_root: str = "") -> dict:
    """Set status on every node in root_id's subtree, children before parents.

    Returns {"updated": <count>}. Validates the whole subtree before writing, so
    illegal transitions leave the graph unchanged.
    """
    target = _parse_todo_status(status)
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(root_id)
        if node is None:
            raise ValueError(f"node {root_id} not found")
        children_map = graph.get_children_map()
        ordered = subtree_post_order(children_map, node)
        for n in ordered:
            validate_todo_status(n, target)
            if target == NodeStatus.DONE:
                assert_done_verified(graph, n)
        updated = 0
        for n in ordered:
            if n.status != target:
                apply_todo_status(graph, n, target)
                updated += 1
        _logger.info(
            "milknado_set_subtree_status: root=%d status=%s updated=%d", root_id, status, updated
        )
        return {"updated": updated}
    finally:
        graph.close()


@mcp.tool()
def milknado_track_follow_up(
    description: str,
    kind: Kind = "task",
    parent_id: int | None = None,
    project_root: str = "",
    files: list[str] | None = None,
    flavor: Flavor | None = None,
    artifact: str | None = None,
    prereqs: list[int] | None = None,
) -> dict:
    """Register discovered follow-up work as a new node.

    Without parent_id, attaches beside the worker's current node; without worker
    context, creates a root node. Optional files record ownership hints. flavor is
    valid only for task nodes. artifact: opaque repo-relative markdown narrative path,
    persisted verbatim. prereqs: node ids that must complete first; rejected on cycle.
    """
    node_kind = _parse_kind(kind)
    root = resolve_project_root(project_root or None)
    _reject_escaping_artifact(artifact, root)
    graph, cfg = open_graph(root)
    try:
        node_flavor = _parse_flavor(flavor, cfg.flavor_registry) if flavor is not None else None
        if parent_id is None:
            parent_id = follow_up_parent_id(graph)
        node = graph.add_node(
            description,
            parent_id=parent_id,
            spec=NodeSpec(
                kind=node_kind,
                flavor=node_flavor,
                artifact_path=artifact,
                prereqs=prereqs or (),
                flavor_registry=cfg.flavor_registry,
            ),
        )
        if files is not None:
            graph.set_file_ownership(node.id, normalize_hint_paths(files, root))
        return node_to_summary(node)
    finally:
        graph.close()


@mcp.tool()
def milknado_delete_node(node_id: int, cascade: bool = False, project_root: str = "") -> dict:
    """Delete a node (coordinator-only). Refuses a node with children unless cascade=True."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        count = graph.delete_node(node_id, cascade=cascade)
        _logger.info(
            "milknado_delete_node: node=%d cascade=%s deleted=%d", node_id, cascade, count
        )
        return {"deleted": count}
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
        _logger.info("milknado_move_node: node=%d -> parent=%s", node_id, new_parent_id)
        return node_to_summary(moved)
    finally:
        graph.close()


@mcp.tool()
def milknado_edit_node(
    node_id: int,
    description: str | None = None,
    kind: Kind | None = None,
    flavor: Flavor | None = None,
    project_root: str = "",
    files: list[str] | None = None,
    artifact: str | None = None,
) -> dict:
    """Edit a node's description, kind, flavor, and/or artifact (coordinator-only).

    Status is not editable. files: None leaves hints untouched; [] clears all hints;
    a non-empty list replaces them. flavor: only valid for task nodes; raises ValueError
    on non-task nodes. artifact: opaque repo-relative markdown narrative path, persisted
    verbatim with no existence check.
    """
    if (
        description is None
        and kind is None
        and flavor is None
        and files is None
        and artifact is None
    ):
        raise ValueError(
            "nothing to edit: provide description, kind, flavor, files, and/or artifact"
        )
    node_kind = _parse_kind(kind) if kind is not None else None
    root = resolve_project_root(project_root or None)
    _reject_escaping_artifact(artifact, root)
    hint_paths = normalize_hint_paths(files, root) if files is not None else None
    graph, cfg = open_graph(root)
    try:
        node_flavor = _parse_flavor(flavor, cfg.flavor_registry) if flavor is not None else None
        if node_flavor is not None and node_kind is not None and node_kind != NodeKind.TASK:
            raise ValueError(
                "flavor is only valid for task nodes; cannot set flavor with non-task kind"
            )
        if node_flavor is not None and node_kind is None:
            # Check existing node kind before setting flavor
            existing = graph.get_node(node_id)
            if existing is None:
                raise ValueError(f"node {node_id} not found")
            if existing.kind != NodeKind.TASK:
                raise ValueError(
                    f"flavor is only valid for task nodes; node {node_id} has kind"
                    f" {existing.kind.value!r}"
                )
        if (
            description is not None
            or node_kind is not None
            or node_flavor is not None
            or artifact is not None
        ):
            graph.update_node(
                node_id,
                description=description,
                kind=node_kind,
                flavor=node_flavor,
                artifact_path=artifact,
                flavor_registry=cfg.flavor_registry,
            )
        if hint_paths is not None:
            node_exists = graph.get_node(node_id) is not None
            only_files = (
                description is None
                and node_kind is None
                and node_flavor is None
                and artifact is None
            )
            if only_files and not node_exists:
                raise ValueError(f"node {node_id} not found")
            graph.set_file_ownership(node_id, hint_paths)
        updated = graph.get_node(node_id)
        if updated is None:
            raise ValueError(f"node {node_id} not found after edit")
        return node_to_summary(updated)
    finally:
        graph.close()
