"""Bounded read-only projection for durable run observers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import msgspec

from milknado.domains.graph._run_persistence import run_row_to_dict

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


@dataclass(frozen=True, slots=True)
class ObserverSnapshot:
    runs: tuple[DurableRun, ...]
    goal: str
    available: int


def _connect(db_path: Path) -> sqlite3.Connection:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ = conn.execute("PRAGMA query_only=ON")
    _ = conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _durable_runs(conn: sqlite3.Connection, limit: int) -> tuple[DurableRun, ...]:
    rows: list[sqlite3.Row] = conn.execute(
        "SELECT r.*, n.description FROM runs r JOIN nodes n ON n.id = r.node_id "
        + "ORDER BY r.started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return tuple(
        msgspec.convert(
            run_row_to_dict(row) | {"description": row["description"]},
            type=DurableRun,
            strict=True,
        )
        for row in rows
    )


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


def read_observer_snapshot(db_path: Path, limit: int = 50) -> ObserverSnapshot:
    """Read bounded observer facts in one transaction without writer maintenance."""
    if not 0 <= limit <= 100:
        raise ValueError("limit must be between 0 and 100")
    conn = _connect(db_path)
    try:
        _ = conn.execute("BEGIN")
        ready_row = cast("sqlite3.Row", conn.execute(_READY_COUNT_SQL).fetchone())
        return ObserverSnapshot(
            runs=_durable_runs(conn, limit),
            goal=_goal_description(conn),
            available=cast(int, ready_row[0]),
        )
    finally:
        conn.rollback()
        conn.close()
