"""Node status state-machine transitions for MikadoGraph.

Free functions taking a connection, mirroring `_persistence.py` / `_mutations.py`.
Every status change is validated against VALID_TRANSITIONS before any write, so
an illegal move raises InvalidTransition rather than corrupting the row.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from milknado.domains.common import VALID_TRANSITIONS, NodeStatus
from milknado.domains.common.errors import InvalidTransition


def assert_transition(conn: sqlite3.Connection, node_id: int, target: NodeStatus) -> NodeStatus:
    """Validate target is reachable from the node's current status; return current."""
    row = conn.execute("SELECT status FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row is None:
        raise ValueError(f"Node {node_id} not found")
    current = NodeStatus(row[0])
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
    params: tuple,
) -> None:
    """Run a CAS UPDATE gated on the source status observed by assert_transition,
    committing then raising InvalidTransition on 0 rows.

    A 0-row result means a concurrent writer moved the node off the status
    assert_transition saw between its SELECT and this UPDATE. Re-read the node's
    actual status so the raised InvalidTransition reports where the node truly is
    now (and the valid_targets reachable from there), not the stale pre-conflict
    value assert_transition observed.
    """
    cur = conn.execute(sql, params)
    conn.commit()
    if cur.rowcount == 0:
        row = conn.execute("SELECT status FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise ValueError(f"Node {node_id} not found")
        actual = NodeStatus(row[0])
        raise InvalidTransition(
            node_id=node_id,
            current=actual,
            target=target,
            valid_targets=tuple(VALID_TRANSITIONS.get(actual, set())),
        )


def transition_status(conn: sqlite3.Connection, node_id: int, target: NodeStatus) -> None:
    """Validate then apply a plain status change (sets completed_at on DONE)."""
    current = assert_transition(conn, node_id, target)
    completed_at = datetime.now(UTC).isoformat() if target == NodeStatus.DONE else None
    _apply_transition(
        conn,
        node_id,
        target,
        "UPDATE nodes SET status = ?, completed_at = ? WHERE id = ? AND status = ?",
        (target.value, completed_at, node_id, current.value),
    )


def mark_failed(conn: sqlite3.Connection, node_id: int) -> None:
    current = assert_transition(conn, node_id, NodeStatus.FAILED)
    _apply_transition(
        conn,
        node_id,
        NodeStatus.FAILED,
        "UPDATE nodes SET status = ?, completed_at = NULL, "
        "worktree_path = NULL, branch_name = NULL, run_id = NULL WHERE id = ? AND status = ?",
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
    _apply_transition(
        conn,
        node_id,
        NodeStatus.RUNNING,
        "UPDATE nodes SET status = ?, completed_at = NULL, "
        "worktree_path = ?, branch_name = ?, run_id = ? WHERE id = ? AND status = ?",
        (NodeStatus.RUNNING.value, worktree_path, branch_name, run_id, node_id, current.value),
    )


def mark_pending(conn: sqlite3.Connection, node_id: int) -> None:
    current = assert_transition(conn, node_id, NodeStatus.PENDING)
    _apply_transition(
        conn,
        node_id,
        NodeStatus.PENDING,
        "UPDATE nodes SET status = ?, completed_at = NULL, "
        "worktree_path = NULL, branch_name = NULL, run_id = NULL WHERE id = ? AND status = ?",
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
        f"UPDATE nodes SET status = 'running', run_id = ?, dispatched_at = ?, pid = ?, "  # noqa: S608 — fixed tuple
        f"worktree_path = NULL, branch_name = NULL WHERE id = ? AND status IN {_CLAIMABLE}",
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
        "worktree_path = NULL, branch_name = NULL, completed_at = NULL "
        "WHERE id = ? AND run_id = ? AND status = 'running'",
        (node_id, owner_run_id),
    )
    conn.commit()
    return cur.rowcount == 1


def mark_terminal(conn: sqlite3.Connection, node_id: int, run_id: str, status: NodeStatus) -> bool:
    """Write a terminal status gated on the active run fence.

    A failed integration retains its worktree and branch so the produced
    deliverable remains recoverable. A later claim clears that recovery state.
    """
    if status is NodeStatus.DONE:
        completed_at = datetime.now(UTC).isoformat()
        cur = conn.execute(
            "UPDATE nodes SET status = ?, completed_at = ? "
            "WHERE id = ? AND run_id = ? AND status = 'running'",
            (NodeStatus.DONE.value, completed_at, node_id, run_id),
        )
    elif status is NodeStatus.FAILED:
        cur = conn.execute(
            "UPDATE nodes SET status = ?, completed_at = NULL, run_id = NULL "
            "WHERE id = ? AND run_id = ? AND status = 'running'",
            (NodeStatus.FAILED.value, node_id, run_id),
        )
    else:
        raise ValueError(f"mark_terminal status must be DONE or FAILED, got {status}")
    conn.commit()
    return cur.rowcount == 1


def mark_blocked(conn: sqlite3.Connection, node_id: int, run_id: str) -> bool:
    """Fence a RUNNING node into BLOCKED without clearing its worktree pin."""
    cur = conn.execute(
        "UPDATE nodes SET status = ?, completed_at = NULL "
        "WHERE id = ? AND run_id = ? AND status = ?",
        (NodeStatus.BLOCKED.value, node_id, run_id, NodeStatus.RUNNING.value),
    )
    conn.commit()
    return cur.rowcount == 1
