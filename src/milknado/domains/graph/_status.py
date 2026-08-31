"""Status-transition free functions for MikadoGraph, driven by StatusPipeline.

Each function wraps the corresponding _transitions.* call (SQL unchanged) in
a pipeline.run(...) so registered StatusMiddleware fire before/after the
write, in the original log-then-notify order.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import cast

import milknado.domains.graph._reads as _reads
import milknado.domains.graph._transitions as _transitions
from milknado.domains.common import MikadoNode, NodeStatus, pid_alive
from milknado.domains.graph._goal_claims import release_goal_claim_on_terminal
from milknado.domains.graph._pipeline import StatusPipeline
from milknado.domains.graph._sqlite_rows import as_tuple as _values
from milknado.domains.graph._sqlite_rows import fetchone
from milknado.domains.graph._verification import validate_done_verification
from milknado.domains.graph.status_flow import (
    subtree_post_order,
    todo_status_steps,
    validate_todo_status,
)

_logger = logging.getLogger(__name__)


def _reject_archived(conn: sqlite3.Connection, node_id: int) -> None:
    """Refuse status writes on an archived node (shelved work is immutable)."""
    row = fetchone(conn, "SELECT archived_at FROM nodes WHERE id = ?", (node_id,))
    if row is not None and _values(row)[0] is not None:
        raise ValueError(f"Node {node_id} is archived; unarchive it first.")


def _apply_subtree_status(
    pipeline: StatusPipeline, conn: sqlite3.Connection, node: MikadoNode, target: NodeStatus
) -> bool:
    if node.status == target:
        return False
    for step in todo_status_steps(node.status, target):
        if step is NodeStatus.PENDING:
            mark_pending(pipeline, conn, node.id)
        else:
            transition_status(pipeline, conn, node.id, step)
        if step is NodeStatus.DONE or step is NodeStatus.FAILED:
            release_goal_claim_on_terminal(conn, node.id)
    return True


def transition_status(
    pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int, target: NodeStatus
) -> None:
    _reject_archived(conn, node_id)
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.transition_status(conn, node_id, target)
        _logger.debug("node %d: %s → %s", node_id, old.value if old else "?", target.value)
        return True

    _ = pipeline.run(lambda nid: _reads.get_node(conn, nid), node_id, old, target, mutate)


def mark_done(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    transition_status(pipeline, conn, node_id, NodeStatus.DONE)
    release_goal_claim_on_terminal(conn, node_id)


def mark_failed(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    _reject_archived(conn, node_id)
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.mark_failed(conn, node_id)
        return True

    _ = pipeline.run(
        lambda nid: _reads.get_node(conn, nid), node_id, old, NodeStatus.FAILED, mutate
    )
    release_goal_claim_on_terminal(conn, node_id)


def mark_running(
    pipeline: StatusPipeline,
    conn: sqlite3.Connection,
    node_id: int,
    worktree_path: str | None = None,
    branch_name: str | None = None,
    run_id: str | None = None,
) -> None:
    _reject_archived(conn, node_id)
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.mark_running(conn, node_id, worktree_path, branch_name, run_id)
        return True

    _ = pipeline.run(
        lambda nid: _reads.get_node(conn, nid), node_id, old, NodeStatus.RUNNING, mutate
    )


def mark_pending(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    _reject_archived(conn, node_id)
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.mark_pending(conn, node_id)
        return True

    _ = pipeline.run(
        lambda nid: _reads.get_node(conn, nid), node_id, old, NodeStatus.PENDING, mutate
    )


def mark_blocked(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    transition_status(pipeline, conn, node_id, NodeStatus.BLOCKED)


def set_todo_status(
    pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int, target: NodeStatus
) -> bool:
    """Validate verification and transition before applying one facade request."""
    node = _reads.get_node(conn, node_id)
    if node is None:
        raise ValueError(f"Node {node_id} not found")
    _reject_archived(conn, node_id)
    if target is NodeStatus.DONE:
        validate_done_verification(conn, node_id, [node])
    validate_todo_status(node, target)
    return _apply_subtree_status(pipeline, conn, node, target)


def set_subtree_status(
    pipeline: StatusPipeline, conn: sqlite3.Connection, root_id: int, target: NodeStatus
) -> int:
    """Validate once, then update each live subtree node children-first."""
    root = _reads.get_node(conn, root_id)
    if root is None:
        raise ValueError(f"Node {root_id} not found")
    ordered = subtree_post_order(
        _reads.get_children_map(conn, include_archived=True),
        root,
    )
    live = [node for node in ordered if node.archived_at is None]
    if target is NodeStatus.DONE:
        validate_done_verification(conn, root_id, live)
    for node in live:
        validate_todo_status(node, target)
    return sum(_apply_subtree_status(pipeline, conn, node, target) for node in live)


def reconcile_completed_goals(pipeline: StatusPipeline, conn: sqlite3.Connection) -> int:
    """Mark pending goals DONE when all direct children are already DONE.

    Repeat because reconciling a child goal can make its parent eligible in the
    same startup pass. Each candidate uses the bulk state-machine operation,
    so its validation happens before any write.
    """
    completed = 0
    while True:
        candidates = [
            node
            for node in _reads.get_all_nodes(conn)
            if node.kind.value == "goal" and node.status is NodeStatus.PENDING
        ]
        eligible = [
            node
            for node in candidates
            if (children := _reads.get_children(conn, node.id))
            and all(child.status is NodeStatus.DONE for child in children)
        ]
        if not eligible:
            return completed
        for goal in eligible:
            completed += set_subtree_status(pipeline, conn, goal.id, NodeStatus.DONE)


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
    *,
    preserve_recovery: bool = False,
) -> bool:
    """Write a terminal status (DONE/FAILED) gated on the run_id fence."""

    def mutate() -> bool:
        return _transitions.mark_terminal(
            conn, node_id, run_id, status, preserve_recovery=preserve_recovery
        )

    ok = pipeline.run(
        lambda nid: _reads.get_node(conn, nid), node_id, NodeStatus.RUNNING, status, mutate
    )
    if ok:
        release_goal_claim_on_terminal(conn, node_id)
    return ok


def mark_blocked_fenced(
    pipeline: StatusPipeline,
    conn: sqlite3.Connection,
    node_id: int,
    run_id: str,
) -> bool:
    """Fence a running node into BLOCKED while retaining its pinned worktree."""
    return pipeline.run(
        lambda nid: _reads.get_node(conn, nid),
        node_id,
        NodeStatus.RUNNING,
        NodeStatus.BLOCKED,
        lambda: _transitions.mark_blocked(conn, node_id, run_id),
    )


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
    # Kept for symmetry with claim_node/mark_terminal's now= convention;
    # pid_alive is the sole signal here.
    _ = now
    row = fetchone(conn, "SELECT status, run_id, pid FROM nodes WHERE id = ?", (node_id,))
    if row is None:
        return False
    values = _values(row)
    if NodeStatus(cast(str, values[0])) != NodeStatus.RUNNING:
        return False
    owner_run_id = cast(str | None, values[1])
    pid = cast(int | None, values[2])
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
