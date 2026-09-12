"""Complete, bounded graph snapshots for run and watch readers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import milknado.domains.graph.snapshot_history as _history
from milknado.domains.common.types import MikadoEdge
from milknado.domains.graph._run_persistence import RunRecord, run_row_to_dict
from milknado.domains.graph.snapshot_models import (
    GraphSnapshot,
    NodeDetailResponse,
    NodeDetailSnapshot,
    SnapshotPage,
)


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ = conn.execute("PRAGMA query_only=ON")
    _ = conn.execute("PRAGMA busy_timeout=5000")
    return conn


def read_graph_snapshot_connection(conn: sqlite3.Connection) -> GraphSnapshot:
    rows = conn.execute("SELECT * FROM nodes WHERE archived_at IS NULL ORDER BY id").fetchall()
    nodes = _history.hydrate(conn, rows)
    node_ids = {node.id for node in nodes}
    edge_rows = cast(
        list[sqlite3.Row],
        conn.execute(
            "SELECT parent_id, child_id FROM edges ORDER BY parent_id, child_id"
        ).fetchall(),
    )
    edges = tuple(
        MikadoEdge(cast(int, row["parent_id"]), cast(int, row["child_id"]))
        for row in edge_rows
        if cast(int, row["parent_id"]) in node_ids and cast(int, row["child_id"]) in node_ids
    )
    children = {edge.child_id for edge in edges}
    return GraphSnapshot(nodes, edges, tuple(node.id for node in nodes if node.id not in children))


def _detail(
    conn: sqlite3.Connection, node_id: int, page: int, limit: int
) -> NodeDetailSnapshot | None:
    node = _history.node(conn, node_id)
    if node is None:
        return None
    ancestors_page = _history.ancestor_page(conn, node, page, limit)
    parent = None if node.parent_id is None else _history.node(conn, node.parent_id)
    if parent is not None and parent.archived_at is not None:
        parent = None
    children = _history.nodes(
        conn,
        "SELECT n.* FROM nodes n JOIN edges e ON e.child_id = n.id WHERE e.parent_id = ? "
        + "AND n.archived_at IS NULL ORDER BY n.id LIMIT ? OFFSET ?",
        "SELECT COUNT(*) FROM edges e JOIN nodes n ON n.id = e.child_id "
        + "WHERE e.parent_id = ? AND n.archived_at IS NULL",
        node_id,
        page,
        limit,
    )
    prerequisite_ids = _history.page(
        conn,
        "SELECT e.child_id AS id FROM edges e JOIN nodes n ON n.id = e.child_id "
        + "WHERE e.parent_id = ? AND n.archived_at IS NULL "
        + "ORDER BY e.child_id LIMIT ? OFFSET ?",
        "SELECT COUNT(*) FROM edges e JOIN nodes n ON n.id = e.child_id "
        + "WHERE e.parent_id = ? AND n.archived_at IS NULL",
        node_id,
        page,
        limit,
        lambda row: cast(int, row["id"]),
    )
    dependent_ids = _history.page(
        conn,
        "SELECT e.parent_id AS id FROM edges e JOIN nodes n ON n.id = e.parent_id "
        + "WHERE e.child_id = ? AND n.archived_at IS NULL "
        + "ORDER BY e.parent_id LIMIT ? OFFSET ?",
        "SELECT COUNT(*) FROM edges e JOIN nodes n ON n.id = e.parent_id "
        + "WHERE e.child_id = ? AND n.archived_at IS NULL",
        node_id,
        page,
        limit,
        lambda row: cast(int, row["id"]),
    )
    reverse_dependents = _history.nodes(
        conn,
        "SELECT n.* FROM nodes n JOIN edges e ON e.parent_id = n.id WHERE e.child_id = ? "
        + "AND n.archived_at IS NULL ORDER BY n.id LIMIT ? OFFSET ?",
        "SELECT COUNT(*) FROM edges e JOIN nodes n ON n.id = e.parent_id "
        + "WHERE e.child_id = ? AND n.archived_at IS NULL",
        node_id,
        page,
        limit,
    )
    owned_files = _history.page(
        conn,
        "SELECT file_path FROM file_ownership WHERE node_id = ? "
        + "ORDER BY file_path LIMIT ? OFFSET ?",
        "SELECT COUNT(*) FROM file_ownership WHERE node_id = ?",
        node_id,
        page,
        limit,
        lambda row: cast(str, row["file_path"]),
        _history.table_exists(conn, "file_ownership"),
    )
    runs = _history.page(
        conn,
        "SELECT * FROM runs WHERE node_id = ? "
        + "ORDER BY started_at DESC, run_id DESC LIMIT ? OFFSET ?",
        "SELECT COUNT(*) FROM runs WHERE node_id = ?",
        node_id,
        page,
        limit,
        run_row_to_dict,
    )
    return NodeDetailSnapshot(
        node=node,
        description=node.description,
        parent=parent,
        children=children,
        ancestors=ancestors_page,
        prerequisite_ids=cast(SnapshotPage[int], prerequisite_ids),
        dependent_ids=cast(SnapshotPage[int], dependent_ids),
        reverse_dependents=reverse_dependents,
        owned_files=cast(SnapshotPage[str], owned_files),
        runs=cast(SnapshotPage[RunRecord], runs),
        reviews=_history.reviews(conn, node_id, page, limit),
        sessions=_history.sessions(conn, node_id, page, limit),
        goal_claim=_history.claim(conn, node),
        artifacts=_history.artifacts(node, page, limit),
    )


def read_node_detail_connection(  # noqa: PLR0913 - response fence and page are one read
    conn: sqlite3.Connection,
    node_id: int,
    request_generation: int = 0,
    page: int = 0,
    limit: int = 50,
) -> NodeDetailResponse:
    _history.validate(page, limit)
    return NodeDetailResponse(node_id, request_generation, _detail(conn, node_id, page, limit))
