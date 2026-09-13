"""Bounded node history projections used by the graph snapshot facade."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar, cast

from milknado.domains.common.session import SessionEvent
from milknado.domains.common.types import MikadoNode, NodeKind
from milknado.domains.graph._goal_claims import GoalClaim
from milknado.domains.graph._run_persistence import NodeReviewRecord
from milknado.domains.graph._session_persistence import event_page, view_session
from milknado.domains.graph.snapshot_models import (
    ArtifactSnapshot,
    NodeSessionSnapshot,
    SnapshotPage,
    SnapshotState,
    SnapshotValue,
)

_T = TypeVar("_T")


def validate(page: int, limit: int) -> None:
    if page < 0 or not 1 <= limit <= 100:
        raise ValueError("page must be non-negative and limit must be between 1 and 100")


def _make_page(
    items: tuple[_T, ...],
    offset: int,
    limit: int,
    total: int | None,
) -> SnapshotPage[_T]:
    return SnapshotPage(
        items,
        offset,
        limit,
        total,
        bool(total is not None and offset + len(items) < total),
    )


def values_page(  # noqa: PLR0913 - page metadata is one immutable contract
    items: list[_T], page: int, limit: int, total: int | None, state: SnapshotState = "loaded"
) -> SnapshotPage[_T]:
    offset = page * limit
    if state != "loaded":
        return SnapshotPage(None, offset, limit, None, False, state)
    return _make_page(tuple(items[offset : offset + limit]), offset, limit, total)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _with_claim(node: MikadoNode, claims: list[sqlite3.Row]) -> MikadoNode:
    for claim in claims:
        if cast(int, claim["goal_id"]) == node.id:
            return replace(node, goal_run_id=cast(str, claim["run_id"]))
    return node


def hydrate(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> tuple[MikadoNode, ...]:
    from milknado.domains.graph._persistence import row_to_node

    nodes = {cast(int, row["id"]): row_to_node(row) for row in rows}
    if not nodes or not table_exists(conn, "goal_claims"):
        return tuple(nodes.values())
    ids = tuple(nodes)
    marks = ",".join("?" for _ in ids)
    claims = conn.execute(
        f"SELECT goal_id, run_id FROM goal_claims WHERE goal_id IN ({marks})", ids
    ).fetchall()
    return tuple(
        _with_claim(node, claims) if node.kind == NodeKind.GOAL else node
        for node in nodes.values()
    )


def node(conn: sqlite3.Connection, node_id: int) -> MikadoNode | None:
    row = cast(
        sqlite3.Row | None,
        conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone(),
    )
    return None if row is None else hydrate(conn, [row])[0]


def page(  # noqa: PLR0913 - one bounded SQL page contract
    conn: sqlite3.Connection,
    rows_sql: str,
    count_sql: str,
    node_id: int,
    page_number: int,
    limit: int,
    mapper: Callable[[sqlite3.Row], object],
    stored: bool = True,
) -> SnapshotPage[object]:
    validate(page_number, limit)
    offset = page_number * limit
    if not stored:
        return SnapshotPage(None, offset, limit, None, False, "not_stored")
    total = cast(int, conn.execute(count_sql, (node_id,)).fetchone()[0])
    rows = cast(list[sqlite3.Row], conn.execute(rows_sql, (node_id, limit, offset)).fetchall())
    return _make_page(tuple(mapper(row) for row in rows), offset, limit, total)


def nodes(  # noqa: PLR0913 - node relation page carries SQL and window
    conn: sqlite3.Connection,
    rows_sql: str,
    count_sql: str,
    node_id: int,
    page_number: int,
    limit: int,
) -> SnapshotPage[MikadoNode]:
    validate(page_number, limit)
    total = cast(int, conn.execute(count_sql, (node_id,)).fetchone()[0])
    offset = page_number * limit
    rows = cast(list[sqlite3.Row], conn.execute(rows_sql, (node_id, limit, offset)).fetchall())
    items = hydrate(conn, rows)
    return _make_page(items, offset, limit, total)


def reviews(
    conn: sqlite3.Connection, node_id: int, page_number: int, limit: int
) -> SnapshotPage[NodeReviewRecord]:
    result = page(
        conn,
        "SELECT node_id, round, verdict, findings, created_at FROM node_reviews "
        + "WHERE node_id = ? ORDER BY round LIMIT ? OFFSET ?",
        "SELECT COUNT(*) FROM node_reviews WHERE node_id = ?",
        node_id,
        page_number,
        limit,
        lambda row: {
            "node_id": cast(int, row["node_id"]),
            "round": cast(int, row["round"]),
            "verdict": cast(str, row["verdict"]),
            "findings": cast(str, row["findings"]),
            "created_at": cast(str, row["created_at"]),
        },
        table_exists(conn, "node_reviews"),
    )
    return cast(SnapshotPage[NodeReviewRecord], result)


def sessions(  # noqa: PLR0913 - node and event pages share one read
    conn: sqlite3.Connection,
    node_id: int,
    page_number: int,
    limit: int,
    event_page_number: int,
) -> SnapshotPage[NodeSessionSnapshot]:
    stored = table_exists(conn, "run_sessions") and table_exists(conn, "run_messages")
    result = page(
        conn,
        "SELECT run_id FROM runs WHERE node_id = ? ORDER BY started_at DESC, run_id DESC "
        + "LIMIT ? OFFSET ?",
        "SELECT COUNT(*) FROM runs WHERE node_id = ?",
        node_id,
        page_number,
        limit,
        lambda row: _session(conn, cast(str, row["run_id"]), stored, event_page_number, limit),
        stored,
    )
    return cast(SnapshotPage[NodeSessionSnapshot], result)


def _empty_event_page(limit: int, state: SnapshotState = "loaded") -> SnapshotPage[SessionEvent]:
    items = None if state != "loaded" else ()
    total = None if state != "loaded" else 0
    return SnapshotPage(items, 0, limit, total, False, state)


def _session(  # noqa: PLR0913 - session identity and page bounds share one read
    conn: sqlite3.Connection,
    run_id: str,
    stored: bool,
    event_page_number: int,
    event_limit: int,
) -> NodeSessionSnapshot:
    if not stored:
        return NodeSessionSnapshot(
            run_id, None, "not_stored", _empty_event_page(event_limit, "not_stored")
        )
    row = cast(
        sqlite3.Row | None,
        conn.execute("SELECT 1 FROM run_sessions WHERE run_id = ?", (run_id,)).fetchone(),
    )
    if row is None:
        return NodeSessionSnapshot(run_id, None, "loaded", _empty_event_page(event_limit))
    session = view_session(conn, run_id)
    history = event_page(conn, run_id, event_page_number, event_limit)
    return NodeSessionSnapshot(run_id, session, "loaded", history)


_ANCESTOR_CTE = """
WITH RECURSIVE ancestor(id, depth, path) AS (
    SELECT parent_id, 1, printf(',%d,', parent_id)
    FROM nodes
    WHERE id = ? AND parent_id IS NOT NULL
    UNION ALL
    SELECT parent.parent_id, ancestor.depth + 1,
           ancestor.path || printf('%d,', parent.parent_id)
    FROM ancestor
    JOIN nodes parent ON parent.id = ancestor.id
    WHERE parent.parent_id IS NOT NULL
      AND instr(ancestor.path, printf(',%d,', parent.parent_id)) = 0
)
"""


def ancestor_page(
    conn: sqlite3.Connection, node_value: MikadoNode, page_number: int, limit: int
) -> SnapshotPage[MikadoNode]:
    validate(page_number, limit)
    total = cast(
        int,
        conn.execute(
            _ANCESTOR_CTE
            + "SELECT COUNT(*) FROM ancestor JOIN nodes n ON n.id = ancestor.id "
            + "WHERE n.archived_at IS NULL",
            (node_value.id,),
        ).fetchone()[0],
    )
    offset = page_number * limit
    rows = cast(
        list[sqlite3.Row],
        conn.execute(
            _ANCESTOR_CTE
            + "SELECT n.* FROM ancestor JOIN nodes n ON n.id = ancestor.id "
            + "WHERE n.archived_at IS NULL ORDER BY ancestor.depth LIMIT ? OFFSET ?",
            (node_value.id, limit, offset),
        ).fetchall(),
    )
    items = hydrate(conn, rows)
    return _make_page(items, offset, limit, total)


def claim(conn: sqlite3.Connection, node_value: MikadoNode) -> SnapshotValue[GoalClaim]:
    if not table_exists(conn, "goal_claims"):
        return SnapshotValue(None, "not_stored")
    if node_value.kind != NodeKind.GOAL:
        return SnapshotValue(None, "loaded")
    row = cast(
        sqlite3.Row | None,
        conn.execute(
            "SELECT goal_id, run_id, pid, claimed_at FROM goal_claims WHERE goal_id = ?",
            (node_value.id,),
        ).fetchone(),
    )
    value: GoalClaim | None = None
    if row is not None:
        value = {
            "goal_id": cast(int, row["goal_id"]),
            "run_id": cast(str, row["run_id"]),
            "pid": cast(int | None, row["pid"]),
            "claimed_at": cast(str, row["claimed_at"]),
        }
    return SnapshotValue(value, "loaded")


def artifacts(
    node_value: MikadoNode, page_number: int, limit: int
) -> SnapshotPage[ArtifactSnapshot]:
    validate(page_number, limit)
    if node_value.artifact_path is None:
        return values_page([], page_number, limit, 0)
    return values_page(
        [ArtifactSnapshot(node_value.artifact_path, SnapshotValue(None, "not_loaded"))],
        page_number,
        limit,
        1,
    )
