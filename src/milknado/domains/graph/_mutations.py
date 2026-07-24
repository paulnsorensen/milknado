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
    """Soft-hide an all-DONE subtree: stamp archived_at on every node in it.

    Eligibility is fail-loud: any non-DONE node in the subtree raises
    ArchiveIneligible naming the offending ids and nothing is stamped.
    Reversible via unarchive_subtree. Returns the number of nodes archived.
    """
    if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
        raise ValueError(f"Node {node_id} not found")
    ordered = _collect_subtree_post_order(children_id_map(conn), node_id)
    placeholders = ",".join("?" * len(ordered))
    rows = conn.execute(
        f"SELECT id, status FROM nodes WHERE id IN ({placeholders})",  # noqa: S608 — placeholders only
        ordered,
    ).fetchall()
    non_done = sorted(row[0] for row in rows if row[1] != NodeStatus.DONE.value)
    if non_done:
        raise ArchiveIneligible(tuple(non_done))
    stamp = now or datetime.now(UTC).isoformat()
    with conn:
        conn.executemany(
            "UPDATE nodes SET archived_at = ? WHERE id = ?",
            [(stamp, nid) for nid in ordered],
        )
    return len(ordered)


def unarchive_subtree(conn: sqlite3.Connection, node_id: int) -> int:
    """Cascade restore: clear archived_at on node_id and every archived descendant.

    Refuses when node_id has any archived ancestor — surfacing a node under a
    still-archived ancestor would break the invariant *archived ⇒ whole
    subtree archived*. Both the ancestor check and the descendant walk run over
    the canonical edge relation (``children_id_map``), the same relation
    ``archive_subtree`` stamps, so diamond wiring (an edge-parent that is not
    the ``nodes.parent_id`` parent) cannot slip an archived ancestor past the
    refusal. Returns the number of nodes un-archived; 0 when neither node_id
    nor any descendant is archived.
    """
    if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
        raise ValueError(f"Node {node_id} not found")
    children_map = children_id_map(conn)
    parents_map: dict[int, list[int]] = {}
    for parent_id, child_ids in children_map.items():
        for child_id in child_ids:
            parents_map.setdefault(child_id, []).append(parent_id)
    archived_ancestors: set[int] = set()
    visited: set[int] = set()
    stack = list(parents_map.get(node_id, []))
    while stack:
        ancestor = stack.pop()
        if ancestor in visited:
            continue
        visited.add(ancestor)
        row = conn.execute("SELECT archived_at FROM nodes WHERE id = ?", (ancestor,)).fetchone()
        if row is None:
            continue
        if row["archived_at"] is not None:
            archived_ancestors.add(ancestor)
        stack.extend(parents_map.get(ancestor, []))
    if archived_ancestors:
        ancestor_id = min(archived_ancestors)
        raise ValueError(
            f"Cannot unarchive {node_id} under archived ancestor {ancestor_id}; "
            "unarchive the ancestor first."
        )
    ordered = _collect_subtree_post_order(children_map, node_id)
    placeholders = ",".join("?" * len(ordered))
    archived = [
        r[0]
        for r in conn.execute(
            f"SELECT id FROM nodes WHERE id IN ({placeholders}) AND archived_at IS NOT NULL",  # noqa: S608 — placeholders only
            ordered,
        )
    ]
    with conn:
        conn.executemany(
            "UPDATE nodes SET archived_at = NULL WHERE id = ?",
            [(nid,) for nid in archived],
        )
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
            raise ValueError(f"re-parenting {node_id} under {new_parent_id} would create a cycle")
    with conn:
        conn.execute("DELETE FROM edges WHERE child_id = ?", (node_id,))
        if new_parent_id is not None:
            conn.execute(
                "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
                (new_parent_id, node_id),
            )
        conn.execute("UPDATE nodes SET parent_id = ? WHERE id = ?", (new_parent_id, node_id))
