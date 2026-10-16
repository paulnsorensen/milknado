"""Fenced release of node ownership."""

from __future__ import annotations

import sqlite3

NO_OPEN_WORKERS = (
    "NOT EXISTS (SELECT 1 FROM run_workers AS w WHERE w.node_id = nodes.id AND w.ended_at IS NULL)"
)


def release(
    conn: sqlite3.Connection, node_id: int, owner_run_id: str, *, reclaim: bool = False
) -> bool:
    """Release a running owner unless bounded unknown-start policy holds it."""
    owns_transaction = not conn.in_transaction
    attempt_guard = (
        " AND NOT EXISTS (SELECT 1 FROM task_attempts AS a WHERE a.node_id = nodes.id "
        + "AND a.owner_run_id = nodes.run_id AND a.outcome = 'active' "
        + "AND a.max_node_attempts > 1)"
        if reclaim
        else ""
    )
    cur = conn.execute(
        "UPDATE nodes SET status = 'pending', run_id = NULL, pid = NULL, "
        + "worktree_path = NULL, branch_name = NULL, completed_at = NULL "
        + "WHERE id = ? AND run_id = ? AND status = 'running' AND "
        + NO_OPEN_WORKERS
        + attempt_guard,
        (node_id, owner_run_id),
    )
    if owns_transaction:
        conn.commit()
    return cur.rowcount == 1
