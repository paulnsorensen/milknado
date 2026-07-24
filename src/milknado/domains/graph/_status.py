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
from milknado.domains.graph._mutations import _collect_subtree_post_order
from milknado.domains.graph._persistence import children_id_map
from milknado.domains.graph._pipeline import StatusPipeline
from milknado.domains.graph.status_flow import todo_status_steps, validate_todo_status

_logger = logging.getLogger(__name__)


def _reject_archived(conn: sqlite3.Connection, node_id: int) -> None:
    """Refuse status writes on an archived node (shelved work is immutable)."""
    row = conn.execute("SELECT archived_at FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row is not None and row[0] is not None:
        raise ValueError(f"Node {node_id} is archived; unarchive it first.")


def transition_status(
    pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int, target: NodeStatus
) -> None:
    _reject_archived(conn, node_id)
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
    _reject_archived(conn, node_id)
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
    _reject_archived(conn, node_id)
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.mark_running(conn, node_id, worktree_path, branch_name, run_id)
        return True

    pipeline.run(lambda nid: _reads.get_node(conn, nid), node_id, old, NodeStatus.RUNNING, mutate)


def mark_pending(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    _reject_archived(conn, node_id)
    old = _reads.node_status(conn, node_id)

    def mutate() -> bool:
        _transitions.mark_pending(conn, node_id)
        return True

    pipeline.run(lambda nid: _reads.get_node(conn, nid), node_id, old, NodeStatus.PENDING, mutate)


def mark_blocked(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    transition_status(pipeline, conn, node_id, NodeStatus.BLOCKED)


def set_subtree_status(
    pipeline: StatusPipeline, conn: sqlite3.Connection, root_id: int, target: NodeStatus
) -> int:
    """Set `target` on every non-archived node in root_id's subtree, children first.

    Archived nodes are SKIPPED — neither stamped nor errored — so bulk
    reconciliation never resurrects shelved work. The live set is validated
    before any write, so an illegal transition leaves the subtree unchanged.
    Returns the number of nodes whose status changed.
    """
    root = _reads.get_node(conn, root_id)
    if root is None:
        raise ValueError(f"Node {root_id} not found")
    ordered = _collect_subtree_post_order(children_id_map(conn), root_id)
    live = []
    for nid in ordered:
        node = _reads.get_node(conn, nid)
        if node is None:
            raise ValueError(f"Node {nid} not found")
        if node.archived_at is None:
            live.append(node)
    for node in live:
        validate_todo_status(node, target)
    updated = 0
    for node in live:
        if node.status == target:
            continue
        for step in todo_status_steps(node.status, target):
            if step is NodeStatus.PENDING:
                # Dispatch through the same field-clearing write as mark_pending:
                # a bare status flip would strand stale worktree/run pins whose
                # run_id a later terminal step could mistake for authoritative.
                mark_pending(pipeline, conn, node.id)
            else:
                transition_status(pipeline, conn, node.id, step)
            if step is NodeStatus.DONE or step is NodeStatus.FAILED:
                # Mirror mark_terminal: a terminal transition releases the goal
                # claim so the sweep and foreign claimants are not fenced out
                # by a row belonging to completed work.
                release_goal_claim_on_terminal(conn, node.id)
        updated += 1
    return updated


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
