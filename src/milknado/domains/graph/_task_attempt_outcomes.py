"""SQL for owner handoff and non-failure outcomes of durable task attempts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from milknado.domains.graph._task_attempt_reads import get_attempt
from milknado.domains.graph._task_attempts import LostAttemptFence

NONFAILURE_OUTCOMES = frozenset(
    {"done", "cancelled", "review_blocked", "preserved", "startup_failed"}
)

_REVIEW_ROUND_FENCE = (
    " AND EXISTS (SELECT 1 FROM runs "
    + "WHERE runs.run_id = ? AND runs.node_id = task_attempts.node_id "
    + "AND runs.attempt_id = task_attempts.attempt_id AND runs.status = 'running')"
)


@dataclass(frozen=True, slots=True)
class OwnerSwap:
    attempt_id: str
    old_owner: str
    new_owner: str
    invocation_id: str


def swap_owner(conn: sqlite3.Connection, swap: OwnerSwap, review_round: bool) -> bool:
    """Swap the node fence and rebind the attempt to the new owner and invocation."""
    attempt_id, old_owner = swap.attempt_id, swap.old_owner
    new_owner, invocation_id = swap.new_owner, swap.invocation_id
    sql = (
        "UPDATE task_attempts SET owner_run_id = ?, active_invocation_id = ?, "
        + "start_state = 'unknown' WHERE attempt_id = ? AND owner_run_id = ? "
        + "AND outcome = 'active'"
    )
    params: tuple[str, ...] = (new_owner, invocation_id, attempt_id, old_owner)
    if review_round:
        sql += _REVIEW_ROUND_FENCE
        params = (*params, invocation_id)
    try:
        with conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            node = conn.execute(
                "UPDATE nodes SET run_id = ? WHERE id = "
                + "(SELECT node_id FROM task_attempts WHERE attempt_id = ?) "
                + "AND status = 'running' AND run_id = ?",
                (new_owner, attempt_id, old_owner),
            )
            if node.rowcount != 1:
                raise LostAttemptFence
            if conn.execute(sql, params).rowcount != 1:
                raise LostAttemptFence
    except LostAttemptFence:
        return False
    return True


def record_nonfailure(
    conn: sqlite3.Connection, attempt_id: str, owner_run_id: str, outcome: str
) -> bool:
    if outcome not in NONFAILURE_OUTCOMES:
        raise ValueError("invalid nonfailure task outcome")
    record = get_attempt(conn, attempt_id)
    if record is None or record.owner_run_id != owner_run_id:
        return False
    if record.outcome == outcome:
        return True
    if record.outcome != "active":
        return False
    cursor = conn.execute(
        "UPDATE task_attempts SET outcome = ? WHERE attempt_id = ? "
        + "AND owner_run_id = ? AND outcome = 'active' "
        + "AND EXISTS (SELECT 1 FROM nodes WHERE nodes.id = task_attempts.node_id "
        + "AND nodes.run_id = ? "
        + "AND nodes.status IN ('running', 'done', 'blocked', 'failed'))",
        (outcome, attempt_id, owner_run_id, owner_run_id),
    )
    conn.commit()
    if cursor.rowcount == 1:
        return True
    settled = get_attempt(conn, attempt_id)
    return (
        settled is not None and settled.owner_run_id == owner_run_id and settled.outcome == outcome
    )
