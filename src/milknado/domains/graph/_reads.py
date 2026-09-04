"""Read-path free functions for MikadoGraph.

Inline SELECT logic extracted from graph.py, mirroring the _persistence.py /
_mutations.py / _transitions.py free-function convention.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import replace
from typing import cast

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus
from milknado.domains.graph._goal_claims import get_goal_claim
from milknado.domains.graph._persistence import children_id_map, row_to_node
from milknado.domains.graph._sqlite_rows import fetchall, fetchone


def _values(row: sqlite3.Row) -> tuple[object, ...]:
    return cast(tuple[object, ...], cast(object, row))


def node_status(conn: sqlite3.Connection, node_id: int) -> NodeStatus | None:
    row = fetchone(conn, "SELECT status FROM nodes WHERE id = ?", (node_id,))
    return NodeStatus(cast(str, _values(row)[0])) if row else None


def get_node(conn: sqlite3.Connection, node_id: int) -> MikadoNode | None:
    """Point lookup by id — not a forest projection.

    An archived node is still returned: callers addressing a node by id
    (dispatch reconcile, briefs, harvest) need the real row. Unlike the
    forest walks below there is deliberately no ``include_archived`` flag —
    a point lookup never filters.
    """
    row = fetchone(conn, "SELECT * FROM nodes WHERE id = ?", (node_id,))
    if row is None:
        return None
    node = row_to_node(row)
    if node.kind == NodeKind.GOAL:
        claim = get_goal_claim(conn, node_id)
        if claim is not None:
            return replace(node, goal_run_id=claim["run_id"])
    return node


def get_nodes(conn: sqlite3.Connection, node_ids: Iterable[int]) -> list[MikadoNode]:
    """Load nodes in input order with duplicate IDs preserved."""
    ids = list(node_ids)
    if not ids:
        return []
    unique_ids = list(dict.fromkeys(ids))
    rows: list[sqlite3.Row] = []
    for start in range(0, len(unique_ids), 500):
        chunk = unique_ids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(fetchall(conn, f"SELECT * FROM nodes WHERE id IN ({placeholders})", chunk))
    nodes = {cast(int, _values(row)[0]): row_to_node(row) for row in rows}
    goal_ids = [node_id for node_id, node in nodes.items() if node.kind == NodeKind.GOAL]
    if goal_ids:
        placeholders = ",".join("?" for _ in goal_ids)
        claims = fetchall(
            conn,
            f"SELECT goal_id, run_id FROM goal_claims WHERE goal_id IN ({placeholders})",
            goal_ids,
        )
        for claim in claims:
            values = _values(claim)
            goal_id = cast(int, values[0])
            run_id = cast(str, values[1])
            nodes[goal_id] = replace(nodes[goal_id], goal_run_id=run_id)
    return [nodes[node_id] for node_id in ids if node_id in nodes]


def latest_results_for_nodes(conn: sqlite3.Connection, node_ids: Iterable[int]) -> dict[int, str]:
    """Return each requested node's newest deposited result body."""
    ids = list(dict.fromkeys(node_ids))
    results: dict[int, str] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = fetchall(
            conn,
            "SELECT r.node_id, m.body "
            + "FROM runs r CROSS JOIN run_messages m "
            + "WHERE m.run_id = r.run_id AND m.role = 'result' "
            + f"AND r.node_id IN ({placeholders}) "
            + "AND NOT EXISTS ("
            + "SELECT 1 FROM runs newer_r CROSS JOIN run_messages newer_m "
            + "WHERE newer_r.node_id = r.node_id "
            + "AND newer_m.run_id = newer_r.run_id AND newer_m.role = 'result' "
            + "AND (newer_m.created_at > m.created_at OR ("
            + "newer_m.created_at = m.created_at AND newer_m.seq > m.seq))"
            + ")",
            chunk,
        )
        results.update(
            (cast(int, values[0]), cast(str, values[1]))
            for values in (_values(row) for row in rows)
        )
    return results


def find_node_by_wiki_ref(conn: sqlite3.Connection, wiki_ref: str) -> MikadoNode | None:
    """Return the node carrying this deterministic wiki key, or None.

    The round-trip lookup the importer/exporter use: wiki_ref is computed
    from git-resident frontmatter, identical across dbs, so it is the stable
    join between a wiki goal file and its milknado node.
    """
    row = fetchone(conn, "SELECT * FROM nodes WHERE wiki_ref = ?", (wiki_ref,))
    return row_to_node(row) if row else None


def find_node_by_github_ref(conn: sqlite3.Connection, github_ref: str) -> MikadoNode | None:
    """Return the node carrying this GitHub Projects node id, or None.

    The stable join between a GitHub project/item and its milknado node:
    github_ref is the opaque PVT/PVTI id, identical across dbs, so it is the
    idempotency key the github importer/binder round-trip on.
    """
    row = fetchone(conn, "SELECT * FROM nodes WHERE github_ref = ?", (github_ref,))
    return row_to_node(row) if row else None


def get_all_nodes(conn: sqlite3.Connection, *, include_archived: bool = False) -> list[MikadoNode]:
    sql = "SELECT * FROM nodes"
    if not include_archived:
        sql += " WHERE archived_at IS NULL"
    return [row_to_node(row) for row in fetchall(conn, sql)]


def get_children(
    conn: sqlite3.Connection, node_id: int, *, include_archived: bool = False
) -> list[MikadoNode]:
    sql = "SELECT n.* FROM nodes n JOIN edges e ON n.id = e.child_id WHERE e.parent_id = ?"
    if not include_archived:
        sql += " AND n.archived_at IS NULL"
    rows = fetchall(conn, sql, (node_id,))
    return [row_to_node(r) for r in rows]


def get_children_map(
    conn: sqlite3.Connection, *, include_archived: bool = False
) -> dict[int, list[MikadoNode]]:
    """Map parent ids to child nodes with archive filtering performed in SQL."""
    mapping: dict[int, list[MikadoNode]] = {}
    if not include_archived:
        rows = fetchall(
            conn,
            """
            SELECT e.parent_id AS edge_parent_id, n.*
            FROM edges e
            JOIN nodes p ON p.id = e.parent_id
            JOIN nodes n ON n.id = e.child_id
            WHERE p.archived_at IS NULL AND n.archived_at IS NULL
            """,
        )
        for row in rows:
            mapping.setdefault(cast(int, _values(row)[0]), []).append(row_to_node(row))
        return mapping
    nodes = {node.id: node for node in get_all_nodes(conn, include_archived=True)}
    for parent_id, child_ids in children_id_map(conn).items():
        children = [nodes[child_id] for child_id in child_ids if child_id in nodes]
        if children:
            mapping[parent_id] = children
    return mapping


def get_leaves(conn: sqlite3.Connection, *, include_archived: bool = False) -> list[MikadoNode]:
    sql = "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT parent_id FROM edges)"
    if not include_archived:
        sql += " AND archived_at IS NULL"
    rows = fetchall(conn, sql)
    return [row_to_node(r) for r in rows]


def get_ready_nodes(
    conn: sqlite3.Connection,
    *,
    kind: NodeKind | None = None,
    flavor: str | None = None,
    limit: int = 100,
    include_archived: bool = False,
) -> list[MikadoNode]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    filters = ["n.status = ?", "EXISTS (SELECT 1 FROM edges i WHERE i.child_id = n.id)"]
    if not include_archived:
        # Defensive: eligibility restricts archive to all-DONE subtrees, which
        # the status filter already excludes, but the read layer must not rely
        # on the mutation layer for its hiding contract.
        filters.append("n.archived_at IS NULL")
    params: list[object] = [NodeStatus.PENDING.value]
    if kind is not None:
        filters.append("n.kind = ?")
        params.append(kind.value)
    if flavor is not None:
        filters.append("n.flavor = ?")
        params.append(flavor)
    params.append(limit)
    rows = fetchall(
        conn,
        "SELECT n.* FROM nodes n WHERE "
        + " AND ".join(filters)
        + " AND NOT EXISTS (SELECT 1 FROM edges e JOIN nodes c ON c.id = e.child_id "
        + "WHERE e.parent_id = n.id AND c.status != 'done') ORDER BY n.id LIMIT ?",
        params,
    )
    return [row_to_node(row) for row in rows]


def get_node_summaries(
    conn: sqlite3.Connection,
    *,
    status: NodeStatus | None = None,
    kind: NodeKind | None = None,
    flavor: str | None = None,
    page: tuple[int, int] = (100, 0),
    include_archived: bool = False,
) -> list[dict[str, int | str]]:
    limit, offset = page
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    filters: list[str] = []
    params: list[object] = []
    if not include_archived:
        filters.append("archived_at IS NULL")
    for column, value in (
        ("status", status.value if status else None),
        ("kind", kind.value if kind else None),
        ("flavor", flavor),
    ):
        if value is not None:
            filters.append(f"{column} = ?")
            params.append(value)
    where = f" WHERE {' AND '.join(filters)}" if filters else ""
    params.extend((limit, offset))
    rows = fetchall(
        conn,
        f"SELECT id, status, description FROM nodes{where} ORDER BY id LIMIT ? OFFSET ?",
        params,
    )
    return [
        {
            "id": cast(int, _values(row)[0]),
            "status": cast(str, _values(row)[1]),
            "description": cast(str, _values(row)[2]),
        }
        for row in rows
    ]


def get_root(conn: sqlite3.Connection, *, include_archived: bool = False) -> MikadoNode | None:
    sql = "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT child_id FROM edges)"
    if not include_archived:
        sql += " AND archived_at IS NULL"
    row = fetchone(conn, sql + " ORDER BY id LIMIT 1")
    return row_to_node(row) if row else None


def get_roots(conn: sqlite3.Connection, *, include_archived: bool = False) -> list[MikadoNode]:
    sql = "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT child_id FROM edges)"
    if not include_archived:
        sql += " AND archived_at IS NULL"
    rows = fetchall(conn, sql)
    return [row_to_node(r) for r in rows]
