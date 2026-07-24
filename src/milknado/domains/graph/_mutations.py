"""Structural node mutations for MikadoGraph: subtree-aware delete and field update.

Free functions taking a connection, mirroring `_persistence.py`. Kept out of
`graph.py` so the facade stays under the file-size ceiling.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from milknado.domains.common import NodeKind, NodeStatus
from milknado.domains.common.errors import ArchiveIneligible, InvalidContainment
from milknado.domains.common.types import BUILTIN_FLAVORS, VALID_CHILD_KINDS
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

    The edges/file_ownership/goal_claims/node_reviews/runs FKs lack ON DELETE
    CASCADE, so dependent rows go first — run_messages before its parent runs
    row, and both
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
    conn.execute("DELETE FROM node_reviews WHERE node_id = ?", (node_id,))
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


def archive_subtree(conn: sqlite3.Connection, node_id: int, *, now: str | None = None) -> int:
    """Atomically validate and archive an all-DONE canonical subtree."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
            raise ValueError(f"Node {node_id} not found")
        rows = conn.execute(
            """
            WITH RECURSIVE subtree(id) AS (
                SELECT ?
                UNION
                SELECT e.child_id FROM edges e JOIN subtree s ON e.parent_id = s.id
            )
            SELECT n.id, n.status FROM nodes n JOIN subtree s ON s.id = n.id
            """,
            (node_id,),
        ).fetchall()
        non_done = sorted(row[0] for row in rows if row[1] != NodeStatus.DONE.value)
        if non_done:
            raise ArchiveIneligible(tuple(non_done))
        stamp = now or datetime.now(UTC).isoformat()
        conn.execute(
            """
            WITH RECURSIVE subtree(id) AS (
                SELECT ?
                UNION
                SELECT e.child_id FROM edges e JOIN subtree s ON e.parent_id = s.id
            )
            UPDATE nodes SET archived_at = ? WHERE id IN (SELECT id FROM subtree)
            """,
            (node_id, stamp),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)


def _external_archived_ancestor(conn: sqlite3.Connection, node_id: int) -> int | None:
    row = conn.execute(
        """
        WITH RECURSIVE subtree(id) AS (
            SELECT ?
            UNION
            SELECT e.child_id FROM edges e JOIN subtree s ON e.parent_id = s.id
        ),
        ancestors(id) AS (
            SELECT e.parent_id FROM edges e JOIN subtree s ON e.child_id = s.id
            UNION
            SELECT e.parent_id FROM edges e JOIN ancestors a ON e.child_id = a.id
        )
        SELECT MIN(a.id)
        FROM ancestors a
        JOIN nodes n ON n.id = a.id
        WHERE n.archived_at IS NOT NULL
          AND a.id NOT IN (SELECT id FROM subtree)
        """,
        (node_id,),
    ).fetchone()
    return row[0]


def _archived_subtree_ids(conn: sqlite3.Connection, node_id: int) -> list[int]:
    rows = conn.execute(
        """
        WITH RECURSIVE subtree(id) AS (
            SELECT ?
            UNION
            SELECT e.child_id FROM edges e JOIN subtree s ON e.parent_id = s.id
        )
        SELECT n.id FROM nodes n JOIN subtree s ON s.id = n.id
        WHERE n.archived_at IS NOT NULL
        """,
        (node_id,),
    ).fetchall()
    return [row[0] for row in rows]


def unarchive_subtree(conn: sqlite3.Connection, node_id: int) -> int:
    """Atomically restore a subtree unless an external archived ancestor remains."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
            raise ValueError(f"Node {node_id} not found")
        ancestor = _external_archived_ancestor(conn, node_id)
        if ancestor is not None:
            raise ValueError(
                f"Cannot unarchive {node_id} under archived ancestor {ancestor}; "
                "unarchive the ancestor first."
            )
        archived = _archived_subtree_ids(conn, node_id)
        conn.executemany(
            "UPDATE nodes SET archived_at = NULL WHERE id = ?",
            [(nid,) for nid in archived],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(archived)


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
    artifact_path: str | None = None,
    flavor_registry: frozenset[str] = BUILTIN_FLAVORS,
) -> None:
    """Update description, kind, flavor, and/or artifact_path; raise if node absent."""
    if flavor is not None and flavor not in flavor_registry:
        raise ValueError(f"unknown flavor {flavor!r}; valid flavors: {sorted(flavor_registry)}")
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
    if artifact_path is not None:
        fields.append("artifact_path = ?")
        values.append(artifact_path)
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
    """Atomically move a node, refusing missing, archived, illegal, or cyclic parents."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        node_row = conn.execute("SELECT kind FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if node_row is None:
            raise ValueError(f"Node {node_id} not found")
        if new_parent_id is not None:
            parent_row = conn.execute(
                "SELECT kind, archived_at FROM nodes WHERE id = ?", (new_parent_id,)
            ).fetchone()
            if parent_row is None:
                raise ValueError(f"new_parent_id {new_parent_id} not found")
            if parent_row["archived_at"] is not None:
                raise ValueError(
                    f"Cannot add a node under archived parent {new_parent_id}; unarchive it first."
                )
            node_kind = NodeKind(node_row["kind"])
            parent_kind = NodeKind(parent_row["kind"])
            if node_kind not in VALID_CHILD_KINDS.get(parent_kind, set()):
                raise InvalidContainment(parent_kind, node_kind)
            if new_parent_id == node_id or would_create_cycle(conn, new_parent_id, node_id):
                raise ValueError(
                    f"re-parenting {node_id} under {new_parent_id} would create a cycle"
                )
        conn.execute("DELETE FROM edges WHERE child_id = ?", (node_id,))
        if new_parent_id is not None:
            conn.execute(
                "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
                (new_parent_id, node_id),
            )
        conn.execute("UPDATE nodes SET parent_id = ? WHERE id = ?", (new_parent_id, node_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
