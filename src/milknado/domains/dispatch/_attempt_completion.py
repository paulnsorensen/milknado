"""Attempt-ending writes shared by the live dispatch paths and reconciliation.

Fence policy: a deleted node is accepted (nothing is left to settle); a lost
attempt or node fence raises, so a stale writer never fails silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph


@dataclass(frozen=True, slots=True)
class AttemptRef:
    node_id: int
    attempt_id: str
    owner_run_id: str
    run_id: str


def settle_counted_failure(graph: MikadoGraph, failure: FinalizeTaskFailure) -> str:
    return str(graph.task_attempts.fail_or_raise(failure).status)


def settle_node_terminal(
    graph: MikadoGraph,
    ref: AttemptRef,
    status: NodeStatus,
    *,
    preserve_recovery: bool = False,
) -> bool:
    """Return False when the node is gone, so callers skip writes that need it."""
    if graph.mark_terminal(ref.node_id, ref.run_id, status, preserve_recovery=preserve_recovery):
        return True
    current = graph.get_node(ref.node_id)
    if current is None:
        return False
    if current.status is status and current.run_id is None:
        return True
    raise RuntimeError(f"terminal node fence lost for run {ref.run_id}")


def settle_preserved(graph: MikadoGraph, ref: AttemptRef, *, record_outcome: bool = True) -> None:
    """Fail the node while keeping the recovery pin of a preserved worktree."""
    if record_outcome:
        graph.task_attempts.require_nonfailure(ref.attempt_id, ref.owner_run_id, "preserved")
    _ = settle_node_terminal(graph, ref, NodeStatus.FAILED, preserve_recovery=True)


def settle_cancelled(graph: MikadoGraph, ref: AttemptRef, finish: RunResult | None = None) -> None:
    """Record the cancelled outcome, then (live path) finish the run, then fail the node."""
    graph.task_attempts.require_nonfailure(ref.attempt_id, ref.owner_run_id, "cancelled")
    if finish is not None:
        graph.runs.finish(ref.run_id, finish)
    _ = settle_node_terminal(graph, ref, NodeStatus.FAILED)
