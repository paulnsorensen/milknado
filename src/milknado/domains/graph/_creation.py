"""Node/edge creation free functions for MikadoGraph.

Inline transaction logic extracted from graph.py, mirroring the
_persistence.py / _mutations.py / _transitions.py free-function convention.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime

from milknado.domains.common import (
    VALID_CHILD_KINDS,
    MikadoEdge,
    MikadoNode,
    NodeKind,
    NodeSpec,
    NodeStatus,
)
from milknado.domains.common.errors import InvalidContainment
from milknado.domains.common.types import DEFAULT_FLAVOR
from milknado.domains.graph._mutations import would_create_cycle
from milknado.domains.graph._persistence import row_to_node
from milknado.domains.graph._reads import get_node

_logger = logging.getLogger(__name__)


def _get_active_parent(conn: sqlite3.Connection, parent_id: int) -> MikadoNode:
    parent = get_node(conn, parent_id)
    if parent is None:
        raise ValueError(f"parent_id {parent_id} not found")
    if parent.archived_at is not None:
        raise ValueError(
            f"Cannot add a node under archived parent {parent_id}; unarchive it first."
        )
    return parent


def _validate_parent(conn: sqlite3.Connection, parent_id: int | None, kind: NodeKind) -> None:
    if parent_id is None:
        return
    parent = _get_active_parent(conn, parent_id)
    if kind not in VALID_CHILD_KINDS.get(parent.kind, set()):
        raise InvalidContainment(parent.kind, kind)


def _validate_prereqs(
    conn: sqlite3.Connection, prereqs: Sequence[int], parent_id: int | None
) -> None:
    if len(prereqs) != len(set(prereqs)):
        raise ValueError("prereqs contains duplicate node ids")
    for prereq_id in prereqs:
        if get_node(conn, prereq_id) is None:
            raise ValueError(f"prereq {prereq_id} not found")
        if prereq_id == parent_id:
            raise ValueError(f"prereq {prereq_id} duplicates parent_id {parent_id}")


def _validate_flavor(
    kind: NodeKind, flavor: str | None, flavor_registry: frozenset[str]
) -> str | None:
    # flavor validation: TASK defaults to IMPLEMENT, non-TASK must be None
    if kind != NodeKind.TASK and flavor is not None:
        raise ValueError(f"flavor must be None for kind={kind.value}; got {flavor!r}")
    flavor_value = (flavor or DEFAULT_FLAVOR) if kind == NodeKind.TASK else None
    if flavor_value is not None and flavor_value not in flavor_registry:
        raise ValueError(
            f"unknown flavor {flavor_value!r}; valid flavors: {sorted(flavor_registry)}"
        )
    return flavor_value


def _insert_node(
    conn: sqlite3.Connection,
    description: str,
    parent_id: int | None,
    spec: NodeSpec,
    flavor: str | None,
) -> int:
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        "INSERT INTO nodes "
        "(description, status, parent_id, created_at, oversized, batch_index, "
        "kind, flavor, wiki_ref, github_ref, artifact_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            description,
            NodeStatus.PENDING.value,
            parent_id,
            now,
            int(spec.oversized),
            spec.batch_index,
            spec.kind.value,
            flavor,
            spec.wiki_ref,
            spec.github_ref,
            spec.artifact_path,
        ),
    )
    if cur.lastrowid is None:
        raise RuntimeError("add_node INSERT did not return lastrowid")
    return cur.lastrowid


def add_node(
    conn: sqlite3.Connection,
    description: str,
    parent_id: int | None,
    spec: NodeSpec,
) -> MikadoNode:
    if not isinstance(spec.kind, NodeKind):
        raise ValueError(f"invalid kind: {spec.kind!r}")
    conn.execute("BEGIN IMMEDIATE")
    try:
        _validate_parent(conn, parent_id, spec.kind)
        _validate_prereqs(conn, spec.prereqs, parent_id)
        flavor = _validate_flavor(spec.kind, spec.flavor, spec.flavor_registry)
        node_id = _insert_node(conn, description, parent_id, spec, flavor)
        if parent_id is not None:
            conn.execute(
                "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
                (parent_id, node_id),
            )
        for prereq_id in spec.prereqs:
            if would_create_cycle(conn, node_id, prereq_id):
                raise ValueError(f"Edge {node_id}->{prereq_id} would create a cycle")
        conn.executemany(
            "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
            [(node_id, prereq_id) for prereq_id in spec.prereqs],
        )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    _logger.debug("node %d created: %r", node_id, description[:80])
    row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return row_to_node(row)


def add_edge(conn: sqlite3.Connection, parent_id: int, child_id: int) -> MikadoEdge:
    conn.execute("BEGIN IMMEDIATE")
    try:
        _get_active_parent(conn, parent_id)
        if would_create_cycle(conn, parent_id, child_id):
            raise ValueError(f"Edge {parent_id}->{child_id} would create a cycle")
        conn.execute(
            "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return MikadoEdge(parent_id=parent_id, child_id=child_id)


def set_batch_metadata(
    conn: sqlite3.Connection, node_id: int, oversized: bool, batch_index: int | None
) -> None:
    cur = conn.execute(
        "UPDATE nodes SET oversized = ?, batch_index = ? WHERE id = ?",
        (1 if oversized else 0, batch_index, node_id),
    )
    if cur.rowcount == 0:
        raise ValueError(f"Node {node_id} not found")
    conn.commit()


def set_parent_id(conn: sqlite3.Connection, node_id: int, parent_id: int | None) -> None:
    """Update parent_id without creating an edge (used by batching bridge)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if parent_id is not None:
            _get_active_parent(conn, parent_id)
        cur = conn.execute(
            "UPDATE nodes SET parent_id = ? WHERE id = ?",
            (parent_id, node_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Node {node_id} not found")
    except Exception:
        conn.rollback()
        raise
    conn.commit()
