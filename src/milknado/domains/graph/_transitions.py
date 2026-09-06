"""Node status state-machine transitions for MikadoGraph.

Free functions taking a connection, mirroring `_persistence.py` / `_mutations.py`.
Every status change is validated against VALID_TRANSITIONS before any write, so
an illegal move raises InvalidTransition rather than corrupting the row.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

from milknado.domains.common import VALID_TRANSITIONS, NodeStatus
from milknado.domains.common.errors import InvalidTransition
from milknado.domains.graph._goal_claims import release_goal_claim_on_terminal
from milknado.domains.graph._sqlite_rows import fetchone


def _values(row: sqlite3.Row) -> tuple[object, ...]:
    return cast(tuple[object, ...], cast(object, row))


def assert_transition(conn: sqlite3.Connection, node_id: int, target: NodeStatus) -> NodeStatus:
    """Validate target is reachable from the node's current status; return current."""
    row = fetchone(conn, "SELECT status FROM nodes WHERE id = ?", (node_id,))
    if row is None:
        raise ValueError(f"Node {node_id} not found")
    current = NodeStatus(cast(str, _values(row)[0]))
    allowed = VALID_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidTransition(
            node_id=node_id,
            current=current,
            target=target,
            valid_targets=tuple(allowed),
        )
    return current


def _apply_transition(
    conn: sqlite3.Connection,
    node_id: int,
    target: NodeStatus,
    sql: str,
    params: Sequence[object],
    *,
    lost_fence_is_noop: bool = False,
) -> bool:
    """Apply a status write and release terminal goal claims at the producer."""
    cur = conn.execute(sql, params)
    conn.commit()
    if cur.rowcount == 0:
        if lost_fence_is_noop:
            return False
        row = fetchone(conn, "SELECT status FROM nodes WHERE id = ?", (node_id,))
        if row is None:
            raise ValueError(f"Node {node_id} not found")
        actual = NodeStatus(cast(str, _values(row)[0]))
        raise InvalidTransition(
            node_id=node_id,
            current=actual,
            target=target,
            valid_targets=tuple(VALID_TRANSITIONS.get(actual, set())),
        )
    if target in (NodeStatus.DONE, NodeStatus.FAILED):
        release_goal_claim_on_terminal(conn, node_id)
    return True


def transition_status(conn: sqlite3.Connection, node_id: int, target: NodeStatus) -> None:
    """Validate then apply a plain status change (sets completed_at on DONE)."""
    current = assert_transition(conn, node_id, target)
    completed_at = datetime.now(UTC).isoformat() if target == NodeStatus.DONE else None
    _ = _apply_transition(
        conn,
        node_id,
        target,
        "UPDATE nodes SET status = ?, completed_at = ? WHERE id = ? AND status = ?",
        (target.value, completed_at, node_id, current.value),
    )


def mark_failed(conn: sqlite3.Connection, node_id: int) -> None:
    current = assert_transition(conn, node_id, NodeStatus.FAILED)
    _ = _apply_transition(
        conn,
        node_id,
        NodeStatus.FAILED,
        "UPDATE nodes SET status = ?, completed_at = NULL, "
        + "worktree_path = NULL, branch_name = NULL, run_id = NULL WHERE id = ? AND status = ?",
        (NodeStatus.FAILED.value, node_id, current.value),
    )


def mark_running(
    conn: sqlite3.Connection,
    node_id: int,
    worktree_path: str | None = None,
    branch_name: str | None = None,
    run_id: str | None = None,
) -> None:
    current = assert_transition(conn, node_id, NodeStatus.RUNNING)
    _ = _apply_transition(
        conn,
        node_id,
        NodeStatus.RUNNING,
        "UPDATE nodes SET status = ?, completed_at = NULL, "
        + "worktree_path = ?, branch_name = ?, run_id = ? WHERE id = ? AND status = ?",
        (NodeStatus.RUNNING.value, worktree_path, branch_name, run_id, node_id, current.value),
    )


def mark_pending(conn: sqlite3.Connection, node_id: int) -> None:
    current = assert_transition(conn, node_id, NodeStatus.PENDING)
    _ = _apply_transition(
        conn,
        node_id,
        NodeStatus.PENDING,
        "UPDATE nodes SET status = ?, completed_at = NULL, "
        + "worktree_path = NULL, branch_name = NULL, run_id = NULL WHERE id = ? AND status = ?",
        (NodeStatus.PENDING.value, node_id, current.value),
    )


# --- Atomic optimistic claim / reclaim / fence ---------------------------------
# These bypass assert_transition deliberately: the SQL WHERE clause IS the guard,
# evaluated atomically by SQLite (a single conditional UPDATE serialized by the
# write lock), which makes it correct across processes — unlike an in-process
# mutex. `cursor.rowcount == 1` tells the caller whether it won.

_CLAIMABLE = ("pending", "failed", "blocked")


def claim_node(
    conn: sqlite3.Connection, node_id: int, run_id: str, now: str, *, pid: int | None = None
) -> bool:
    """Atomically claim a claimable node, including its dispatch PID fence."""
    cur = conn.execute(
        "UPDATE nodes SET status = 'running', run_id = ?, dispatched_at = ?, pid = ?, "
        + f"worktree_path = NULL, branch_name = NULL WHERE id = ? AND status IN {_CLAIMABLE}",
        (run_id, now, pid, node_id),
    )
    conn.commit()
    return cur.rowcount == 1


def release(conn: sqlite3.Connection, node_id: int, owner_run_id: str) -> bool:
    """Flip a RUNNING node back to PENDING, clearing ownership, gated on run_id.

    Used by try_reclaim to free a provably-dead owner and by dispatch cleanup to
    release a claim whose startup failed. The run_id guard means a node already
    re-claimed under a different run is left untouched.

    The `status = 'running'` guard mirrors mark_terminal: DONE keeps its run_id, so
    without it an owner that committed DONE between a reclaim's SELECT and this
    UPDATE could be walked back from DONE to PENDING, resurrecting a completed node.
    """
    cur = conn.execute(
        "UPDATE nodes SET status = 'pending', run_id = NULL, pid = NULL, "
        + "worktree_path = NULL, branch_name = NULL, completed_at = NULL "
        + "WHERE id = ? AND run_id = ? AND status = 'running'",
        (node_id, owner_run_id),
    )
    conn.commit()
    return cur.rowcount == 1


def mark_terminal(
    conn: sqlite3.Connection,
    node_id: int,
    run_id: str,
    status: NodeStatus,
    *,
    preserve_recovery: bool = False,
) -> bool:
    """Write a terminal status gated on the active run fence."""
    if status is NodeStatus.DONE:
        completed_at = datetime.now(UTC).isoformat()
        sql = (
            "UPDATE nodes SET status = ?, completed_at = ? "
            + "WHERE id = ? AND run_id = ? AND status = 'running'"
        )
        params: Sequence[object] = (NodeStatus.DONE.value, completed_at, node_id, run_id)
    elif status is NodeStatus.FAILED:
        recovery = (
            "run_id = NULL"
            if preserve_recovery
            else ("worktree_path = NULL, branch_name = NULL, run_id = NULL")
        )
        sql = (
            f"UPDATE nodes SET status = ?, completed_at = NULL, {recovery} "
            + "WHERE id = ? AND run_id = ? AND status = 'running'"
        )
        params = (NodeStatus.FAILED.value, node_id, run_id)
    else:
        raise ValueError(f"mark_terminal status must be DONE or FAILED, got {status}")
    return _apply_transition(conn, node_id, status, sql, params, lost_fence_is_noop=True)


def mark_blocked(conn: sqlite3.Connection, node_id: int, run_id: str) -> bool:
    """Fence a RUNNING node into BLOCKED without clearing its worktree pin."""
    cur = conn.execute(
        "UPDATE nodes SET status = ?, completed_at = NULL "
        + "WHERE id = ? AND run_id = ? AND status = ?",
        (NodeStatus.BLOCKED.value, node_id, run_id, NodeStatus.RUNNING.value),
    )
    conn.commit()
    return cur.rowcount == 1
