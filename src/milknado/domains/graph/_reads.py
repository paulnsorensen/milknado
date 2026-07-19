"""Read-path free functions for MikadoGraph.

Inline SELECT logic extracted from graph.py, mirroring the _persistence.py /
_mutations.py / _transitions.py free-function convention.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import replace

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus
from milknado.domains.graph._persistence import children_id_map, get_goal_claim, row_to_node


def node_status(conn: sqlite3.Connection, node_id: int) -> NodeStatus | None:
    row = conn.execute("SELECT status FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return NodeStatus(row[0]) if row else None


def get_node(conn: sqlite3.Connection, node_id: int) -> MikadoNode | None:
    row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
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
    rows = []
    for start in range(0, len(unique_ids), 500):
        chunk = unique_ids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            conn.execute(
                f"SELECT * FROM nodes WHERE id IN ({placeholders})",  # noqa: S608
                chunk,
            ).fetchall()
        )
    nodes = {row["id"]: row_to_node(row) for row in rows}
    goal_ids = [node_id for node_id, node in nodes.items() if node.kind == NodeKind.GOAL]
    if goal_ids:
        placeholders = ",".join("?" for _ in goal_ids)
        claims = conn.execute(
            f"SELECT goal_id, run_id FROM goal_claims WHERE goal_id IN ({placeholders})",  # noqa: S608
            goal_ids,
        ).fetchall()
        for goal_id, run_id in claims:
            nodes[goal_id] = replace(nodes[goal_id], goal_run_id=run_id)
    return [nodes[node_id] for node_id in ids if node_id in nodes]


def latest_results_for_nodes(conn: sqlite3.Connection, node_ids: Iterable[int]) -> dict[int, str]:
    """Return each requested node's newest deposited result body."""
    ids = list(dict.fromkeys(node_ids))
    results: dict[int, str] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT node_id, body FROM ("
            "SELECT r.node_id, m.body, ROW_NUMBER() OVER ("
            "PARTITION BY r.node_id ORDER BY m.created_at DESC, m.seq DESC"
            ") AS ordinal FROM runs r JOIN run_messages m ON m.run_id = r.run_id "
            f"WHERE m.role = 'result' AND r.node_id IN ({placeholders})"  # noqa: S608
            ") WHERE ordinal = 1",
            chunk,
        ).fetchall()
        results.update((row["node_id"], row["body"]) for row in rows)
    return results


def find_node_by_wiki_ref(conn: sqlite3.Connection, wiki_ref: str) -> MikadoNode | None:
    """Return the node carrying this deterministic wiki key, or None.

    The round-trip lookup the importer/exporter use: wiki_ref is computed
    from git-resident frontmatter, identical across dbs, so it is the stable
    join between a wiki goal file and its milknado node.
    """
    row = conn.execute("SELECT * FROM nodes WHERE wiki_ref = ?", (wiki_ref,)).fetchone()
    return row_to_node(row) if row else None


def find_node_by_github_ref(conn: sqlite3.Connection, github_ref: str) -> MikadoNode | None:
    """Return the node carrying this GitHub Projects node id, or None.

    The stable join between a GitHub project/item and its milknado node:
    github_ref is the opaque PVT/PVTI id, identical across dbs, so it is the
    idempotency key the github importer/binder round-trip on.
    """
    row = conn.execute("SELECT * FROM nodes WHERE github_ref = ?", (github_ref,)).fetchone()
    return row_to_node(row) if row else None


def get_all_nodes(conn: sqlite3.Connection) -> list[MikadoNode]:
    return [row_to_node(r) for r in conn.execute("SELECT * FROM nodes").fetchall()]


def get_children(conn: sqlite3.Connection, node_id: int) -> list[MikadoNode]:
    rows = conn.execute(
        "SELECT n.* FROM nodes n JOIN edges e ON n.id = e.child_id WHERE e.parent_id = ?",
        (node_id,),
    ).fetchall()
    return [row_to_node(r) for r in rows]


def get_children_map(conn: sqlite3.Connection) -> dict[int, list[MikadoNode]]:
    """Map parent_id -> child nodes, reusing the persistence id-scan.

    Lets callers materialise an entire subtree without issuing one
    get_children query per node (the N+1 pattern).
    """
    nodes = {n.id: n for n in get_all_nodes(conn)}
    mapping: dict[int, list[MikadoNode]] = {}
    for parent_id, child_ids in children_id_map(conn).items():
        kids = [nodes[cid] for cid in child_ids if cid in nodes]
        if kids:
            mapping[parent_id] = kids
    return mapping


def get_leaves(conn: sqlite3.Connection) -> list[MikadoNode]:
    rows = conn.execute(
        "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT parent_id FROM edges)"
    ).fetchall()
    return [row_to_node(r) for r in rows]


def get_ready_nodes(
    conn: sqlite3.Connection,
    *,
    kind: NodeKind | None = None,
    flavor: str | None = None,
    limit: int = 100,
) -> list[MikadoNode]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    filters = ["n.status = ?", "EXISTS (SELECT 1 FROM edges i WHERE i.child_id = n.id)"]
    params: list[object] = [NodeStatus.PENDING.value]
    if kind is not None:
        filters.append("n.kind = ?")
        params.append(kind.value)
    if flavor is not None:
        filters.append("n.flavor = ?")
        params.append(flavor)
    params.append(limit)
    rows = conn.execute(
        "SELECT n.* FROM nodes n WHERE "
        + " AND ".join(filters)
        + " AND NOT EXISTS (SELECT 1 FROM edges e JOIN nodes c ON c.id = e.child_id "
        "WHERE e.parent_id = n.id AND c.status != 'done') ORDER BY n.id LIMIT ?",
        params,
    ).fetchall()
    return [row_to_node(row) for row in rows]


def get_node_summaries(
    conn: sqlite3.Connection,
    *,
    status: NodeStatus | None = None,
    kind: NodeKind | None = None,
    flavor: str | None = None,
    page: tuple[int, int] = (100, 0),
) -> list[dict[str, int | str]]:
    limit, offset = page
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    filters, params = [], []
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
    rows = conn.execute(
        f"SELECT id, status, description FROM nodes{where} ORDER BY id LIMIT ? OFFSET ?",  # noqa: S608
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def get_root(conn: sqlite3.Connection) -> MikadoNode | None:
    row = conn.execute(
        "SELECT * FROM nodes"
        " WHERE id NOT IN (SELECT DISTINCT child_id FROM edges)"
        " ORDER BY id LIMIT 1"
    ).fetchone()
    return row_to_node(row) if row else None


def get_roots(conn: sqlite3.Connection) -> list[MikadoNode]:
    rows = conn.execute(
        "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT child_id FROM edges)"
    ).fetchall()
    return [row_to_node(r) for r in rows]
