"""Bind protected worker invocation to its current task-attempt seed."""

from __future__ import annotations

import sqlite3

_BIND_WORKER_SQL = (
    "UPDATE task_attempts SET active_invocation_id = ?, "
    + "start_state = CASE WHEN active_invocation_id = ? THEN start_state "
    + "ELSE 'unknown' END WHERE attempt_id = ? AND owner_run_id = ? "
    + "AND outcome = 'active' AND (active_invocation_id IN (?, ?) "
    + "OR (active_invocation_id = ? AND ? = "
    + "(SELECT latest.run_id FROM runs AS latest "
    + "WHERE latest.attempt_id = task_attempts.attempt_id "
    + "AND latest.node_id = task_attempts.node_id "
    + "ORDER BY latest.rowid DESC LIMIT 1)) "
    + "OR EXISTS (SELECT 1 FROM run_workers AS previous "
    + "WHERE previous.invocation_id = task_attempts.active_invocation_id "
    + "AND previous.graph_run_id = ? AND previous.node_id = task_attempts.node_id "
    + "AND previous.ended_at IS NOT NULL)) "
    + "AND NOT EXISTS (SELECT 1 FROM run_workers AS live "
    + "WHERE live.node_id = task_attempts.node_id AND live.ended_at IS NULL "
    + "AND live.invocation_id != ?) "
    + "AND EXISTS (SELECT 1 FROM nodes WHERE nodes.id = task_attempts.node_id "
    + "AND nodes.status = 'running' AND nodes.run_id = ?) "
    + "AND EXISTS (SELECT 1 FROM runs WHERE runs.run_id = ? "
    + "AND runs.node_id = task_attempts.node_id AND runs.attempt_id = ?)"
)


def bind_worker_invocation(conn: sqlite3.Connection, identity: tuple[str, str, str, str]) -> bool:
    attempt_id, owner_run_id, source_run_id, invocation_id = identity
    if not invocation_id:
        raise ValueError("invocation identity must be nonempty")
    cursor = conn.execute(
        _BIND_WORKER_SQL,
        (
            invocation_id,
            invocation_id,
            attempt_id,
            owner_run_id,
            source_run_id,
            invocation_id,
            owner_run_id,
            source_run_id,
            source_run_id,
            invocation_id,
            owner_run_id,
            source_run_id,
            attempt_id,
        ),
    )
    conn.commit()
    return cursor.rowcount == 1
