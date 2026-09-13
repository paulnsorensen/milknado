"""Atomic worker follow-up creation with durable provenance."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import milknado.domains.graph._creation as _creation
from milknado.domains.common import MikadoNode, NodeSpec
from milknado.domains.graph._command_persistence import get_capabilities
from milknado.domains.graph._mutations import would_create_cycle
from milknado.domains.graph._persistence import row_to_node
from milknado.domains.graph._sqlite_rows import as_tuple, fetchone


@dataclass(frozen=True)
class FollowUpSource:
    node_id: int
    run_id: str
    invocation_id: str
    request_id: str


@dataclass(frozen=True)
class FollowUpRequest:
    description: str
    parent_id: int | None
    spec: NodeSpec
    files: tuple[str, ...]
    source: FollowUpSource


def _existing_node(conn: sqlite3.Connection, source: FollowUpSource) -> MikadoNode | None:
    row = fetchone(
        conn,
        "SELECT n.* FROM follow_up_provenance p "
        + "JOIN nodes n ON n.id = p.node_id "
        + "WHERE p.source_run_id = ? AND p.source_invocation_id = ? AND p.request_id = ?",
        (source.run_id, source.invocation_id, source.request_id),
    )
    return row_to_node(row) if row is not None else None


def _validate_source(conn: sqlite3.Connection, source: FollowUpSource) -> None:
    run = fetchone(conn, "SELECT node_id, status FROM runs WHERE run_id = ?", (source.run_id,))
    if run is None:
        raise ValueError(f"source run {source.run_id!r} not found")
    run_values = as_tuple(run)
    run_node_id, status = cast(int, run_values[0]), cast(str, run_values[1])
    if run_node_id != source.node_id or status != "running":
        raise ValueError(f"source run {source.run_id!r} is not active for node {source.node_id}")
    capabilities = get_capabilities(conn, source.run_id)
    if (
        capabilities is None
        or capabilities.node_id != source.node_id
        or capabilities.invocation_id != source.invocation_id
    ):
        raise ValueError("follow-up source does not match the active worker invocation")


def _resolve_parent(conn: sqlite3.Connection, request: FollowUpRequest) -> int | None:
    source_node = fetchone(
        conn, "SELECT parent_id FROM nodes WHERE id = ?", (request.source.node_id,)
    )
    if source_node is None:
        raise ValueError(f"source node {request.source.node_id} not found")
    sibling_parent = cast(int | None, as_tuple(source_node)[0])
    return request.parent_id if request.parent_id is not None else sibling_parent


def _insert_edges(conn: sqlite3.Connection, node_id: int, prereqs: tuple[int, ...]) -> None:
    for prereq_id in prereqs:
        if would_create_cycle(conn, node_id, prereq_id):
            raise ValueError(f"Edge {node_id}->{prereq_id} would create a cycle")
    _ = conn.executemany(
        "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
        [(node_id, prereq_id) for prereq_id in prereqs],
    )


def _insert_provenance(conn: sqlite3.Connection, node_id: int, source: FollowUpSource) -> None:
    _ = conn.execute(
        "INSERT INTO follow_up_provenance "
        + "(node_id, source_node_id, source_run_id, source_invocation_id, "
        + "request_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            node_id,
            source.node_id,
            source.run_id,
            source.invocation_id,
            source.request_id,
            datetime.now(UTC).isoformat(),
        ),
    )


def _create_new(conn: sqlite3.Connection, request: FollowUpRequest) -> MikadoNode:
    parent_id = _resolve_parent(conn, request)
    _creation.validate_parent(conn, parent_id, request.spec.kind)
    _creation.validate_prereqs(conn, request.spec.prereqs, parent_id)
    flavor = _creation.validate_flavor(
        request.spec.kind, request.spec.flavor, request.spec.flavor_registry
    )
    node_id = _creation.insert_node(conn, request.description, parent_id, request.spec, flavor)
    if parent_id is not None:
        _ = conn.execute(
            "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)", (parent_id, node_id)
        )
    _insert_edges(conn, node_id, request.spec.prereqs)
    _ = conn.executemany(
        "INSERT INTO file_ownership (node_id, file_path) VALUES (?, ?)",
        [(node_id, path) for path in request.files],
    )
    _insert_provenance(conn, node_id, request.source)
    row = fetchone(conn, "SELECT * FROM nodes WHERE id = ?", (node_id,))
    if row is None:
        raise RuntimeError("follow-up INSERT did not return a node")
    return row_to_node(row)


def create_follow_up(
    conn: sqlite3.Connection, request: FollowUpRequest
) -> tuple[MikadoNode, bool]:
    """Create one worker follow-up, or return its prior idempotent result."""
    _ = conn.execute("BEGIN IMMEDIATE")
    try:
        _validate_source(conn, request.source)
        existing = _existing_node(conn, request.source)
        if existing is not None:
            conn.commit()
            return existing, False
        node = _create_new(conn, request)
        conn.commit()
        return node, True
    except Exception:
        conn.rollback()
        raise
