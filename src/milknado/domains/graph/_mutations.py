"""Structural node mutations for MikadoGraph: subtree-aware delete and field update.

Free functions taking a connection, mirroring `_persistence.py`. Kept out of
`graph.py` so the facade stays under the file-size ceiling.
"""

from __future__ import annotations

import sqlite3

from milknado.domains.common import NodeKind


def _children_map(conn: sqlite3.Connection) -> dict[int, list[int]]:
    """parent_id -> child_ids from a single edges scan (avoids per-node queries)."""
    mapping: dict[int, list[int]] = {}
    for parent_id, child_id in conn.execute("SELECT parent_id, child_id FROM edges").fetchall():
        mapping.setdefault(parent_id, []).append(child_id)
    return mapping


def _collect_subtree_post_order(children_map: dict[int, list[int]], node_id: int) -> list[int]:
    """Subtree ids in post-order (children before parents), each visited once.

    Dedups shared descendants reachable via diamond wiring so a delete pass
    touches every node exactly once and never re-visits an already-removed id.
    """
    visited: set[int] = set()
    ordered: list[int] = []

    def visit(nid: int) -> None:
        if nid in visited:
            return
        visited.add(nid)
        for child in children_map.get(nid, []):
            visit(child)
        ordered.append(nid)

    visit(node_id)
    return ordered


def _delete_one(conn: sqlite3.Connection, node_id: int) -> None:
    """Delete one node and every row referencing it, without committing.

    The edges/file_ownership FKs lack ON DELETE CASCADE, so dependent rows go
    first. `nodes.parent_id` carries no FK, so a `set_parent_id` link (no edge)
    would otherwise dangle once its target is gone — null those references here.
    """
    conn.execute("DELETE FROM edges WHERE parent_id = ? OR child_id = ?", (node_id, node_id))
    conn.execute("DELETE FROM file_ownership WHERE node_id = ?", (node_id,))
    conn.execute("UPDATE nodes SET parent_id = NULL WHERE parent_id = ?", (node_id,))
    conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))


def delete_subtree(conn: sqlite3.Connection, node_id: int, cascade: bool) -> int:
    """Delete a node (and, with cascade, its whole subtree) atomically.

    A node with edge-children is refused unless cascade=True. The cascade runs
    inside one transaction (single commit) so a mid-delete failure rolls the
    whole subtree back rather than leaving it half-removed. Returns the count
    of nodes deleted.
    """
    if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
        raise ValueError(f"Node {node_id} not found")
    children_map = _children_map(conn)
    if children_map.get(node_id) and not cascade:
        raise ValueError(f"Node {node_id} has children; pass cascade=True to delete subtree")
    ordered = _collect_subtree_post_order(children_map, node_id)
    with conn:
        for nid in ordered:
            _delete_one(conn, nid)
    return len(ordered)


def update_node_fields(
    conn: sqlite3.Connection,
    node_id: int,
    description: str | None,
    kind: NodeKind | None,
) -> None:
    """Update description and/or kind on a node; raise if it does not exist."""
    fields: list[str] = []
    values: list[object] = []
    if description is not None:
        fields.append("description = ?")
        values.append(description)
    if kind is not None:
        fields.append("kind = ?")
        values.append(kind.value)
    if not fields:
        raise ValueError("nothing to update")
    values.append(node_id)
    cur = conn.execute(f"UPDATE nodes SET {', '.join(fields)} WHERE id = ?", values)
    if cur.rowcount == 0:
        raise ValueError(f"Node {node_id} not found")
    conn.commit()
