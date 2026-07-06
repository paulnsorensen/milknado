"""Structural node mutations for MikadoGraph: subtree-aware delete and field update.

Free functions taking a connection, mirroring `_persistence.py`. Kept out of
`graph.py` so the facade stays under the file-size ceiling.
"""

from __future__ import annotations

import sqlite3

from milknado.domains.common import NodeKind
from milknado.domains.common.errors import InvalidContainment
from milknado.domains.common.types import VALID_CHILD_KINDS
from milknado.domains.graph._persistence import children_id_map


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

    The edges/file_ownership/goal_claims/runs FKs lack ON DELETE CASCADE, so
    dependent rows go first — run_messages before its parent runs row, and both
    before the node whose runs.node_id FK would otherwise abort the delete.
    `nodes.parent_id` carries no FK, so a `set_parent_id` link (no edge) would
    otherwise dangle once its target is gone — null those references here.
    """
    conn.execute("DELETE FROM edges WHERE parent_id = ? OR child_id = ?", (node_id, node_id))
    conn.execute("DELETE FROM goal_claims WHERE goal_id = ?", (node_id,))
    conn.execute("DELETE FROM file_ownership WHERE node_id = ?", (node_id,))
    conn.execute(
        "DELETE FROM run_messages WHERE run_id IN (SELECT run_id FROM runs WHERE node_id = ?)",
        (node_id,),
    )
    conn.execute("DELETE FROM runs WHERE node_id = ?", (node_id,))
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
    has_children = (
        conn.execute("SELECT 1 FROM edges WHERE parent_id = ? LIMIT 1", (node_id,)).fetchone()
        is not None
    )
    if has_children and not cascade:
        raise ValueError(f"Node {node_id} has children; pass cascade=True to delete subtree")
    if not has_children:
        with conn:
            _delete_one(conn, node_id)
        return 1
    ordered = _collect_subtree_post_order(children_id_map(conn), node_id)
    with conn:
        for nid in ordered:
            _delete_one(conn, nid)
    return len(ordered)


def _validate_kind_change_containment(
    conn: sqlite3.Connection, node_id: int, new_kind: NodeKind
) -> None:
    """Reject a kind change that would create an illegal parent->child pair.

    Mirrors the containment guard in ``reparent``/``add_node`` so editing a
    node's kind cannot bypass ``VALID_CHILD_KINDS`` via its existing edges.
    """
    parent = conn.execute(
        "SELECT n.kind FROM edges e JOIN nodes n ON n.id = e.parent_id WHERE e.child_id = ?",
        (node_id,),
    ).fetchone()
    if parent is not None:
        parent_kind = NodeKind(parent["kind"])
        if new_kind not in VALID_CHILD_KINDS.get(parent_kind, set()):
            raise InvalidContainment(parent_kind, new_kind)
    children = conn.execute(
        "SELECT n.kind FROM edges e JOIN nodes n ON n.id = e.child_id WHERE e.parent_id = ?",
        (node_id,),
    ).fetchall()
    for child in children:
        child_kind = NodeKind(child["kind"])
        if child_kind not in VALID_CHILD_KINDS.get(new_kind, set()):
            raise InvalidContainment(new_kind, child_kind)


def update_node_fields(
    conn: sqlite3.Connection,
    node_id: int,
    description: str | None,
    kind: NodeKind | None,
    flavor: str | None = None,
) -> None:
    """Update description, kind, and/or flavor on a node; raise if it does not exist."""
    fields: list[str] = []
    values: list[object] = []
    if description is not None:
        fields.append("description = ?")
        values.append(description)
    if kind is not None:
        _validate_kind_change_containment(conn, node_id, kind)
        fields.append("kind = ?")
        values.append(kind.value)
        if kind != NodeKind.TASK:
            if flavor is not None:
                raise ValueError(
                    f"flavor must be None for kind={kind.value} nodes; "
                    "cannot set a non-None flavor on a non-task node"
                )
            # Changing to non-task: clear flavor to preserve the invariant
            fields.append("flavor = ?")
            values.append(None)
    if flavor is not None:
        fields.append("flavor = ?")
        values.append(flavor)
    if not fields:
        raise ValueError("nothing to update")
    values.append(node_id)
    cur = conn.execute(f"UPDATE nodes SET {', '.join(fields)} WHERE id = ?", values)
    if cur.rowcount == 0:
        raise ValueError(f"Node {node_id} not found")
    conn.commit()


def would_create_cycle(conn: sqlite3.Connection, parent_id: int, child_id: int) -> bool:
    """True if edge parent_id->child_id would make child_id its own ancestor.

    Walks ancestors of parent_id; reaching child_id means child_id already sits
    above parent_id, so adding the edge closes a loop.
    """
    visited: set[int] = set()
    stack = [parent_id]
    while stack:
        current = stack.pop()
        if current == child_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        rows = conn.execute(
            "SELECT parent_id FROM edges WHERE child_id = ?", (current,)
        ).fetchall()
        stack.extend(row[0] for row in rows)
    return False


def reparent(conn: sqlite3.Connection, node_id: int, new_parent_id: int | None) -> None:
    """Move node_id under new_parent_id (or to root level when None).

    Rewrites the incoming edge and the parent_id column atomically. Rejects a
    missing node or new parent, and a new parent that is the node itself or one
    of its descendants (which would create a cycle).

    Assumes the single-parent tree model: every incoming edge is replaced, so a
    node given multiple parents via add_edge collapses to the one new parent.
    Multi-parent re-parenting is out of scope.
    """
    node_row = conn.execute("SELECT kind FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if node_row is None:
        raise ValueError(f"Node {node_id} not found")
    if new_parent_id is not None:
        parent_row = conn.execute(
            "SELECT kind FROM nodes WHERE id = ?", (new_parent_id,)
        ).fetchone()
        if parent_row is None:
            raise ValueError(f"new_parent_id {new_parent_id} not found")
        node_kind = NodeKind(node_row["kind"])
        parent_kind = NodeKind(parent_row["kind"])
        if node_kind not in VALID_CHILD_KINDS.get(parent_kind, set()):
            raise InvalidContainment(parent_kind, node_kind)
        if new_parent_id == node_id or would_create_cycle(conn, new_parent_id, node_id):
            raise ValueError(f"re-parenting {node_id} under {new_parent_id} would create a cycle")
    with conn:
        conn.execute("DELETE FROM edges WHERE child_id = ?", (node_id,))
        if new_parent_id is not None:
            conn.execute(
                "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
                (new_parent_id, node_id),
            )
        conn.execute("UPDATE nodes SET parent_id = ? WHERE id = ?", (new_parent_id, node_id))
