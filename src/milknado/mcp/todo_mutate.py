"""Milknado MCP todo mutate tools — write tools registered against the shared FastMCP instance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import (
    NodeKind,
    NodeSpec,
    normalize_hint_paths,
    validate_hint_path,
)
from milknado.domains.graph import apply_todo_status
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
from milknado.mcp.todo import follow_up_parent_id, node_to_summary

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CreateTodo:
    description: str
    kind: Kind
    files: list[str] | None
    flavor: Flavor | None
    artifact: str | None
    prereqs: list[int] | None
    root: Path


def _create_todo(graph, cfg, parent_id: int | None, request: _CreateTodo) -> dict:  # noqa: ANN001
    if request.artifact is not None:
        validate_hint_path(request.artifact, request.root, label="artifact")
    files = (
        normalize_hint_paths(request.files, request.root) if request.files is not None else None
    )
    flavor = (
        _parse_flavor(request.flavor, cfg.flavor_registry) if request.flavor is not None else None
    )
    node = graph.add_node(
        request.description,
        parent_id=parent_id,
        spec=NodeSpec(
            kind=_parse_kind(request.kind),
            flavor=flavor,
            artifact_path=request.artifact,
            prereqs=request.prereqs or (),
            flavor_registry=cfg.flavor_registry,
        ),
    )
    if files is not None:
        graph.set_file_ownership(node.id, files)
    return node_to_summary(node)


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

    files: optional paths this node will touch, confined to the project root.
    flavor: a built-in flavor or a registry name declared in milknado.toml;
    validation uses the resolved project flavor registry and is valid only for tasks.
    artifact: opaque repo-relative markdown path, persisted without an existence check.
    prereqs: node ids that must complete before this node is ready.
    """
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        request = _CreateTodo(description, kind, files, flavor, artifact, prereqs, root)
        return _create_todo(graph, cfg, parent_id, request)
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
    """Set status on every live node in root_id's subtree, children before parents.

    Archived nodes are skipped — bulk reconciliation never resurrects shelved
    work. Returns {"updated": <count>}. Validates the whole live set before
    writing, so illegal transitions leave the graph unchanged.
    """
    target = _parse_todo_status(status)
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(root_id)
        if node is None:
            raise ValueError(f"node {root_id} not found")
        updated = graph.set_subtree_status(root_id, target)
        _logger.info(
            "milknado_set_subtree_status: root=%d status=%s updated=%d", root_id, status, updated
        )
        return {"updated": updated}
    finally:
        graph.close()


@mcp.tool()
def milknado_archive_node(node_id: int, project_root: str = "") -> dict:
    """Archive an all-DONE subtree (soft-hide, reversible via unarchive).

    Fails loud naming the live offenders when any node in the subtree is not
    done. Returns {"archived": <count>}.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        if graph.get_node(node_id) is None:
            raise ValueError(f"node {node_id} not found")
        archived = graph.archive_subtree(node_id)
        _logger.info("milknado_archive_node: node=%d archived=%d", node_id, archived)
        return {"archived": archived}
    finally:
        graph.close()


@mcp.tool()
def milknado_unarchive_node(node_id: int, project_root: str = "") -> dict:
    """Restore an archived subtree; refuses while an ancestor stays archived.

    Returns {"unarchived": <count>}.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        if graph.get_node(node_id) is None:
            raise ValueError(f"node {node_id} not found")
        unarchived = graph.unarchive_subtree(node_id)
        _logger.info("milknado_unarchive_node: node=%d unarchived=%d", node_id, unarchived)
        return {"unarchived": unarchived}
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
    context, creates a root node. Flavor accepts a built-in or a registry name
    declared in milknado.toml and is valid only for task nodes.
    """
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        resolved_parent = parent_id if parent_id is not None else follow_up_parent_id(graph)
        request = _CreateTodo(description, kind, files, flavor, artifact, prereqs, root)
        return _create_todo(graph, cfg, resolved_parent, request)
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
    if artifact is not None:
        validate_hint_path(artifact, root, label="artifact")
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
