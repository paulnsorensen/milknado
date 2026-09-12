"""Bounded read-only projection for durable run observers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import msgspec

from milknado.domains.common.session import SessionView
from milknado.domains.graph._run_persistence import run_row_to_dict
from milknado.domains.graph._session_persistence import view_session
from milknado.domains.graph.snapshot_models import GraphSnapshot, NodeDetailResponse

_READY_COUNT_SQL = """
WITH ready(id) AS (
    SELECT n.id
    FROM nodes n
    WHERE n.status = 'pending'
      AND n.archived_at IS NULL
      AND EXISTS (SELECT 1 FROM edges incoming WHERE incoming.child_id = n.id)
      AND NOT EXISTS (
          SELECT 1
          FROM edges outgoing
          JOIN nodes child ON child.id = outgoing.child_id
          WHERE outgoing.parent_id = n.id AND child.status != 'done'
      )
    ORDER BY n.id
    LIMIT 100
)
SELECT COUNT(*)
FROM ready candidate
WHERE NOT EXISTS (
    SELECT 1
    FROM file_ownership candidate_file
    JOIN file_ownership blocker_file ON blocker_file.file_path = candidate_file.file_path
    JOIN nodes blocker ON blocker.id = blocker_file.node_id
    WHERE candidate_file.node_id = candidate.id
      AND blocker.archived_at IS NULL
      AND (
          blocker.status = 'running'
          OR blocker_file.node_id IN (
              SELECT prior.id FROM ready prior WHERE prior.id < candidate.id
          )
      )
)
"""


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


def read_graph_snapshot_connection(conn: sqlite3.Connection) -> GraphSnapshot:
    from milknado.domains.graph.snapshot import read_graph_snapshot_connection as read_graph

    return read_graph(conn)


def read_node_detail_connection(  # noqa: PLR0913 - response fence and page are one read
    conn: sqlite3.Connection,
    node_id: int,
    request_generation: int = 0,
    page: int = 0,
    limit: int = 50,
) -> NodeDetailResponse:
    from milknado.domains.graph.snapshot import read_node_detail_connection as read_detail

    return read_detail(conn, node_id, request_generation, page, limit)


def read_node_detail_snapshot(  # noqa: PLR0913 - response fence and page are one read
    db_path: Path,
    node_id: int,
    request_generation: int = 0,
    page: int = 0,
    limit: int = 50,
) -> NodeDetailResponse:
    from milknado.domains.graph.snapshot import read_node_detail_snapshot as read_detail

    return read_detail(db_path, node_id, request_generation, page, limit)


def read_observer_snapshot(  # noqa: PLR0913 - observer and detail fences share one transaction
    db_path: Path,
    limit: int = 50,
    *,
    node_id: int | None = None,
    request_generation: int = 0,
    page: int = 0,
) -> ObserverSnapshot:
    """Read bounded observer facts in one transaction without writer maintenance."""
    if not 0 <= limit <= 100:
        raise ValueError("limit must be between 0 and 100")
    from milknado.domains.graph.snapshot import (
        connect_readonly,
        read_graph_snapshot_connection,
        read_node_detail_connection,
    )

    conn = connect_readonly(db_path)
    try:
        _ = conn.execute("BEGIN")
        ready_row = cast("sqlite3.Row", conn.execute(_READY_COUNT_SQL).fetchone())
        return ObserverSnapshot(
            runs=_durable_runs(conn, limit),
            goal=_goal_description(conn),
            available=cast(int, ready_row[0]),
            graph=read_graph_snapshot_connection(conn),
            node=(
                read_node_detail_connection(conn, node_id, request_generation, page, limit)
                if node_id is not None
                else None
            ),
        )
    finally:
        conn.rollback()
        conn.close()


__all__ = [
    "DurableRun",
    "GraphSnapshot",
    "NodeDetailResponse",
    "ObserverSnapshot",
    "read_node_detail_snapshot",
    "read_observer_snapshot",
]
