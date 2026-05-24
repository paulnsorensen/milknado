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
