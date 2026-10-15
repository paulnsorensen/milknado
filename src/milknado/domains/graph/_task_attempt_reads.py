"""Read-only queries over durable task attempts."""

from __future__ import annotations

import sqlite3
from typing import cast

from milknado.domains.graph._sqlite_rows import fetchone
from milknado.domains.graph._task_attempt_models import TaskAttemptRecord


def _record(row: sqlite3.Row) -> TaskAttemptRecord:
    return TaskAttemptRecord(
        cast(int, row["node_id"]),
        cast(str, row["attempt_id"]),
        cast(str, row["owner_run_id"]),
        cast(str, row["active_invocation_id"]),
        cast(int, row["max_node_attempts"]),
        cast(str, row["start_state"]),
        cast(str, row["outcome"]),
        cast(int | None, row["failed_count"]),
        cast(str | None, row["disposition"]),
        cast(str | None, row["reason"]),
    )


def get_attempt(conn: sqlite3.Connection, attempt_id: str) -> TaskAttemptRecord | None:
    row = fetchone(conn, "SELECT * FROM task_attempts WHERE attempt_id = ?", (attempt_id,))
    return _record(row) if row is not None else None


def failed_count(conn: sqlite3.Connection, node_id: int) -> int:
    row = fetchone(
        conn,
        "SELECT count(*) FROM task_attempts WHERE node_id = ? AND outcome = 'failed'",
        (node_id,),
    )
    return cast(int, row[0]) if row is not None else 0
