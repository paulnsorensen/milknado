"""Status-transition free functions for MikadoGraph, driven by StatusPipeline.

Each function wraps the corresponding _transitions.* call (SQL unchanged) in
a pipeline.run(...) so registered StatusMiddleware fire before/after the
write, in the original log-then-notify order.
"""

from __future__ import annotations

import logging
import sqlite3

from milknado.domains.common import NodeStatus, pid_alive
from milknado.domains.graph import _reads, _transitions
from milknado.domains.graph._goal_claims import release_goal_claim_on_terminal
from milknado.domains.graph._pipeline import StatusPipeline

_logger = logging.getLogger(__name__)


def transition_status(
    pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int, target: NodeStatus
) -> None:
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.transition_status(conn, node_id, target)
        _logger.debug("node %d: %s → %s", node_id, old.value if old else "?", target.value)
        return True

    pipeline.run(lambda nid: _reads.get_node(conn, nid), node_id, old, target, mutate)


def mark_done(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    transition_status(pipeline, conn, node_id, NodeStatus.DONE)
    release_goal_claim_on_terminal(conn, node_id)


def mark_failed(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.mark_failed(conn, node_id)
        return True

    pipeline.run(lambda nid: _reads.get_node(conn, nid), node_id, old, NodeStatus.FAILED, mutate)
    release_goal_claim_on_terminal(conn, node_id)


def mark_running(
    pipeline: StatusPipeline,
    conn: sqlite3.Connection,
    node_id: int,
    worktree_path: str | None = None,
    branch_name: str | None = None,
    run_id: str | None = None,
) -> None:
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.mark_running(conn, node_id, worktree_path, branch_name, run_id)
        return True

    pipeline.run(lambda nid: _reads.get_node(conn, nid), node_id, old, NodeStatus.RUNNING, mutate)


def mark_pending(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.mark_pending(conn, node_id)
        return True

    pipeline.run(lambda nid: _reads.get_node(conn, nid), node_id, old, NodeStatus.PENDING, mutate)


def mark_blocked(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    transition_status(pipeline, conn, node_id, NodeStatus.BLOCKED)


def complete_root(pipeline: StatusPipeline, conn: sqlite3.Connection) -> bool:
    """Auto-complete root when all non-root nodes are done.

    Returns True if root was completed.
    """
    root = _reads.get_root(conn)
    if root is None or root.status != NodeStatus.PENDING:
        return False
    all_nodes = _reads.get_all_nodes(conn)
    non_root = [n for n in all_nodes if n.id != root.id]
    if not all(n.status == NodeStatus.DONE for n in non_root):
        return False
    mark_running(pipeline, conn, root.id)
    mark_done(pipeline, conn, root.id)
    return True


def claim_node(
    pipeline: StatusPipeline,
    conn: sqlite3.Connection,
    node_id: int,
    run_id: str,
    *,
    now: str,
    pid: int | None = None,
) -> bool:
    """Atomically claim a claimable node, including its dispatch PID fence."""
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        return _transitions.claim_node(conn, node_id, run_id, now, pid=pid)

    return pipeline.run(
        lambda nid: _reads.get_node(conn, nid), node_id, old, NodeStatus.RUNNING, mutate
    )


def release(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int, run_id: str) -> bool:
    """Release a RUNNING claim back to PENDING, fenced on the run_id.

    Used by dispatch cleanup to undo a claim whose startup failed, without
    clobbering a node already re-claimed under a different run. Fenced on both
    run_id and status = 'running' (see _transitions.release), so a node that
    completed DONE between the caller's read and this write is never walked
    back to PENDING. Returns whether the release landed.
    """

    def mutate() -> bool:
        return _transitions.release(conn, node_id, run_id)

    return pipeline.run(
        lambda nid: _reads.get_node(conn, nid),
        node_id,
        NodeStatus.RUNNING,
        NodeStatus.PENDING,
        mutate,
    )


def mark_terminal(
    pipeline: StatusPipeline,
    conn: sqlite3.Connection,
    node_id: int,
    run_id: str,
    status: NodeStatus,
) -> bool:
    """Write a terminal status (DONE/FAILED) gated on the run_id fence.

    Returns False when the node was re-claimed under a new run_id — the caller
    was reclaimed and its terminal write is rejected (zero rows).
    """

    def mutate() -> bool:
        return _transitions.mark_terminal(conn, node_id, run_id, status)

    ok = pipeline.run(
        lambda nid: _reads.get_node(conn, nid), node_id, NodeStatus.RUNNING, status, mutate
    )
    if ok:
        release_goal_claim_on_terminal(conn, node_id)
    return ok


def try_reclaim(
    pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int, *, now: str
) -> bool:
    """Free a RUNNING node whose owner process is provably dead.

    Reads the current owner (run_id, pid); if the pid is recorded and no longer
    alive, releases the node back to PENDING so the next dispatch can claim it
    without waiting out the stale-running timeout. A live (or pid-unknown) owner
    is left intact — a failed claim refuses the new caller instead.

    Returns True iff a dead owner was actually released, so the dispatch boundary
    can log the recovery (an operationally interesting event on the worker path).
    """
    row = conn.execute("SELECT status, run_id, pid FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row is None or NodeStatus(row["status"]) != NodeStatus.RUNNING:
        return False
    owner_run_id, pid = row["run_id"], row["pid"]
    if owner_run_id is None or pid is None or pid_alive(pid):
        return False

    def mutate() -> bool:
        return _transitions.release(conn, node_id, owner_run_id)

    return pipeline.run(
        lambda nid: _reads.get_node(conn, nid),
        node_id,
        NodeStatus.RUNNING,
        NodeStatus.PENDING,
        mutate,
    )
