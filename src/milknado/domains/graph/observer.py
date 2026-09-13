"""Bounded read-only projection for durable run observers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import msgspec

import milknado.domains.graph._persistence as _persistence
import milknado.domains.graph._reads as _reads
from milknado.domains.common.session import SessionView
from milknado.domains.graph._run_persistence import run_row_to_dict
from milknado.domains.graph._session_persistence import view_session
from milknado.domains.graph._sqlite_rows import fetchall
from milknado.domains.graph.snapshot import (
    connect_readonly,
    read_graph_snapshot_connection,
    read_node_detail_connection,
)
from milknado.domains.graph.snapshot_models import GraphSnapshot, NodeDetailResponse

_NODE_DETAIL_DEFAULT_LIMIT = 50


class DurableRun(msgspec.Struct, frozen=True):
    run_id: str
    node_id: int
    description: str
    status: Literal["running", "done", "failed"]
    pid: int | None
    log_path: str
    started_at: str
    ended_at: str | None
    timed_out: bool
    exit_code: int | None
    error: str | None
    timeout_seconds: int | None
    detail: str | None
    rebased: bool | None
    session: SessionView = SessionView()


@dataclass(frozen=True, slots=True)
class ObserverSnapshot:
    runs: tuple[DurableRun, ...]
    goal: str
    available: int
    graph: GraphSnapshot | None = None
    node: NodeDetailResponse | None = None
    graph_revision: int | None = None


def _durable_run(conn: sqlite3.Connection, row: sqlite3.Row) -> DurableRun:
    record = run_row_to_dict(row)
    session = view_session(conn, record["run_id"], active=record["status"] == "running")
    return DurableRun(
        run_id=record["run_id"],
        node_id=record["node_id"],
        description=cast(str, row["description"]),
        status=cast(Literal["running", "done", "failed"], record["status"]),
        pid=record["pid"],
        log_path=record["log_path"],
        started_at=record["started_at"],
        ended_at=record["ended_at"],
        timed_out=record["timed_out"],
        exit_code=record["exit_code"],
        error=record["error"],
        timeout_seconds=record["timeout_seconds"],
        detail=record["detail"],
        rebased=record["rebased"],
        session=session,
    )


def _durable_runs(conn: sqlite3.Connection, limit: int) -> tuple[DurableRun, ...]:
    rows: list[sqlite3.Row] = conn.execute(
        "SELECT r.*, n.description FROM runs r JOIN nodes n ON n.id = r.node_id "
        + "ORDER BY r.started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return tuple(_durable_run(conn, row) for row in rows)


def _goal_description(conn: sqlite3.Connection) -> str:
    row = cast(
        "sqlite3.Row | None",
        conn.execute(
            "SELECT description FROM nodes "
            + "WHERE id NOT IN (SELECT DISTINCT child_id FROM edges) "
            + "AND archived_at IS NULL ORDER BY id LIMIT 1"
        ).fetchone(),
    )
    return row[0] if row is not None else ""


def _available_count(conn: sqlite3.Connection) -> int:
    ready_ids = _reads.get_ready_node_ids(conn)
    if not ready_ids:
        return 0
    running_ids = [
        cast(int, row[0])
        for row in fetchall(conn, "SELECT id FROM nodes WHERE status = 'running'")
    ]
    conflicts = _persistence.check_parallel_safety(conn, [*running_ids, *ready_ids])
    ready_set = set(ready_ids)
    running_set = set(running_ids)
    blocked_ids = {
        right_id
        for left_id, right_id, _ in conflicts
        if right_id in ready_set and (left_id in running_set or left_id in ready_set)
    }
    return len(ready_ids) - len(blocked_ids)


def read_observer_node_snapshot(  # noqa: PLR0913 - node response fence and page share one read
    db_path: Path,
    node_id: int,
    request_generation: int = 0,
    page: int = 0,
    limit: int = _NODE_DETAIL_DEFAULT_LIMIT,
    session_event_page: int = 0,
) -> NodeDetailResponse:
    conn = connect_readonly(db_path)
    try:
        _ = conn.execute("BEGIN")
        return read_node_detail_connection(
            conn, node_id, request_generation, page, limit, session_event_page
        )
    finally:
        conn.rollback()
        conn.close()


def _graph_revision(conn: sqlite3.Connection) -> int:
    row = cast(
        sqlite3.Row, conn.execute("SELECT revision FROM graph_revision WHERE id = 1").fetchone()
    )
    return cast(int, row[0])


def read_observer_snapshot_connection(  # noqa: PLR0913 - observer and detail fences share one transaction
    conn: sqlite3.Connection,
    limit: int = 50,
    *,
    node_id: int | None = None,
    request_generation: int = 0,
    page: int = 0,
    node_limit: int = _NODE_DETAIL_DEFAULT_LIMIT,
    session_event_page: int = 0,
    cached_graph: GraphSnapshot | None = None,
    cached_graph_revision: int | None = None,
) -> ObserverSnapshot:
    if not 0 <= limit <= 100:
        raise ValueError("limit must be between 0 and 100")
    _ = conn.execute("BEGIN")
    try:
        available = _available_count(conn)
        revision = _graph_revision(conn)
        graph = (
            cached_graph
            if cached_graph is not None and revision == cached_graph_revision
            else read_graph_snapshot_connection(conn)
        )
        node = (
            read_node_detail_connection(
                conn, node_id, request_generation, page, node_limit, session_event_page
            )
            if node_id is not None
            else None
        )
        return ObserverSnapshot(
            runs=_durable_runs(conn, limit),
            goal=_goal_description(conn),
            available=available,
            graph=graph,
            node=node,
            graph_revision=revision,
        )
    finally:
        conn.rollback()


def read_observer_snapshot(  # noqa: PLR0913 - observer and detail fences share one transaction
    db_path: Path,
    limit: int = 50,
    *,
    node_id: int | None = None,
    request_generation: int = 0,
    page: int = 0,
    node_limit: int = _NODE_DETAIL_DEFAULT_LIMIT,
    session_event_page: int = 0,
    cached_graph: GraphSnapshot | None = None,
    cached_graph_revision: int | None = None,
    connection: sqlite3.Connection | None = None,
) -> ObserverSnapshot:
    conn = connection or connect_readonly(db_path)
    try:
        return read_observer_snapshot_connection(
            conn,
            limit,
            node_id=node_id,
            request_generation=request_generation,
            page=page,
            node_limit=node_limit,
            session_event_page=session_event_page,
            cached_graph=cached_graph,
            cached_graph_revision=cached_graph_revision,
        )
    finally:
        if connection is None:
            conn.close()


__all__ = [
    "DurableRun",
    "GraphSnapshot",
    "NodeDetailResponse",
    "ObserverSnapshot",
    "read_observer_node_snapshot",
    "read_observer_snapshot",
    "read_observer_snapshot_connection",
]
