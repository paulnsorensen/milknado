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


def assert_transition(conn: sqlite3.Connection, node_id: int, target: NodeStatus) -> None:
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


def transition_status(conn: sqlite3.Connection, node_id: int, target: NodeStatus) -> None:
    """Validate then apply a plain status change (sets completed_at on DONE)."""
    assert_transition(conn, node_id, target)
    completed_at = datetime.now(UTC).isoformat() if target == NodeStatus.DONE else None
    conn.execute(
        "UPDATE nodes SET status = ?, completed_at = ? WHERE id = ?",
        (target.value, completed_at, node_id),
    )
    conn.commit()


def mark_failed(conn: sqlite3.Connection, node_id: int) -> None:
    assert_transition(conn, node_id, NodeStatus.FAILED)
    conn.execute(
        "UPDATE nodes SET status = ?, completed_at = NULL, "
        "worktree_path = NULL, branch_name = NULL, run_id = NULL WHERE id = ?",
        (NodeStatus.FAILED.value, node_id),
    )
    conn.commit()


def mark_running(
    conn: sqlite3.Connection,
    node_id: int,
    worktree_path: str | None = None,
    branch_name: str | None = None,
    run_id: str | None = None,
) -> None:
    assert_transition(conn, node_id, NodeStatus.RUNNING)
    conn.execute(
        "UPDATE nodes SET status = ?, completed_at = NULL, "
        "worktree_path = ?, branch_name = ?, run_id = ? WHERE id = ?",
        (NodeStatus.RUNNING.value, worktree_path, branch_name, run_id, node_id),
    )
    conn.commit()


def mark_pending(conn: sqlite3.Connection, node_id: int) -> None:
    assert_transition(conn, node_id, NodeStatus.PENDING)
    conn.execute(
        "UPDATE nodes SET status = ?, completed_at = NULL, "
        "worktree_path = NULL, branch_name = NULL, run_id = NULL WHERE id = ?",
        (NodeStatus.PENDING.value, node_id),
    )
    conn.commit()


# --- Atomic optimistic claim / reclaim / fence ---------------------------------
# These bypass assert_transition deliberately: the SQL WHERE clause IS the guard,
# evaluated atomically by SQLite (a single conditional UPDATE serialized by the
# write lock), which makes it correct across processes — unlike an in-process
# mutex. `cursor.rowcount == 1` tells the caller whether it won.

_CLAIMABLE = ("pending", "failed", "blocked")


def claim_node(conn: sqlite3.Connection, node_id: int, run_id: str, now: str) -> bool:
    """Atomically claim a claimable node, returning True iff this caller won.

    The PENDING/FAILED/BLOCKED -> RUNNING transition, the run_id fence, and the
    dispatch timestamp are written in one statement; a concurrent caller (another
    thread OR another process sharing the db) sees RUNNING and gets rowcount 0.

    `pid` is reset to NULL: a FAILED node carries the dead prior run's pid
    (mark_terminal clears run_id but not pid), so without this reset a freshly
    re-claimed node would advertise a stale-and-dead pid in the window before
    set_pid records the real one — and a concurrent try_reclaim would read that
    dead pid and release this legitimate claim. NULL means "pid-unknown", which
    try_reclaim correctly refuses to reclaim.
    """
    cur = conn.execute(
        f"UPDATE nodes SET status = 'running', run_id = ?, dispatched_at = ?, pid = NULL "  # noqa: S608 — fixed tuple
        f"WHERE id = ? AND status IN {_CLAIMABLE}",
        (run_id, now, node_id),
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
    """Write a node's terminal status (DONE/FAILED), gated on the run_id fence.

    Returns False (zero rows) when the node has been re-claimed under a new
    run_id — the caller was reclaimed and must NOT treat its run as authoritative.
    DONE sets completed_at and keeps the worktree/branch (already removed on disk);
    FAILED clears worktree/branch/run_id, mirroring mark_failed.

    The `status = 'running'` guard makes the transition fire exactly once from the
    active owner: DONE keeps its run_id, so without it a later same-run_id
    mark_terminal(..., FAILED) would walk a DONE node back to FAILED, bypassing
    the terminal state machine.
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
            "UPDATE nodes SET status = ?, completed_at = NULL, worktree_path = NULL, "
            "branch_name = NULL, run_id = NULL WHERE id = ? AND run_id = ? AND status = 'running'",
            (NodeStatus.FAILED.value, node_id, run_id),
        )
    else:
        raise ValueError(f"mark_terminal status must be DONE or FAILED, got {status}")
    conn.commit()
    return cur.rowcount == 1
