"""Execution-slice operations for one logical node attempt."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from milknado.domains.common import (
    VALID_TRANSITIONS,
    InvalidTransition,
    MikadoNode,
    NodeStatus,
    RunResult,
    UnlandedWorkError,
)
from milknado.domains.dispatch import make_run_id
from milknado.domains.execution._node_context import ExecutionConfig, NodeExecutionContext
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph

_logger = logging.getLogger(__name__)


def dispatch_identity(
    graph: MikadoGraph, node: MikadoNode, config: ExecutionConfig, parent_run_id: str | None
) -> tuple[str, str, str | None, bool]:
    adopted = parent_run_id is not None
    selected_run_id = make_run_id(node.id) if config.max_node_attempts > 1 else None
    attempt_id = uuid4().hex
    if adopted:
        parent_row = graph.runs.get(parent_run_id)
        if parent_row is not None and parent_row["attempt_id"] is not None:
            attempt_id = str(parent_row["attempt_id"])
        if node.status is not NodeStatus.RUNNING or node.run_id != parent_run_id:
            raise InvalidTransition(
                node.id,
                node.status,
                NodeStatus.RUNNING,
                tuple(VALID_TRANSITIONS[node.status]),
            )
        return parent_run_id, attempt_id, selected_run_id, True
    return selected_run_id or make_run_id(node.id), attempt_id, selected_run_id, False


def bind_review_round(
    graph: MikadoGraph, context: NodeExecutionContext, old_run_id: str, run_id: str
) -> None:
    attempt_id = context.attempt_id
    assert attempt_id is not None
    old_owner = context.owner_fence or old_run_id
    new_owner = context.owner_fence or run_id
    if not graph.task_attempts.rotate_review_round(attempt_id, old_owner, new_owner, run_id):
        raise ValueError(f"review redispatch fence lost for attempt {attempt_id}")


def record_nonfailure(
    graph: MikadoGraph, node: MikadoNode, context: NodeExecutionContext | None, outcome: str
) -> None:
    if context is None or context.attempt_id is None or node.run_id is None:
        return
    graph.task_attempts.require_nonfailure(context.attempt_id, node.run_id, outcome)


def discard_worktree(
    remove: Callable[[int, Path], None], node: MikadoNode, reason: str
) -> str | None:
    if not node.worktree_path:
        return None
    wt = Path(node.worktree_path)
    if not wt.exists():
        return None
    try:
        remove(node.id, wt)
    except UnlandedWorkError as exc:
        _logger.warning(
            "Preserving %s node %d worktree %s; recording for recovery: %s",
            reason,
            node.id,
            wt,
            exc,
        )
        return str(wt)
    return None


@dataclass(frozen=True)
class AttemptRuntime:
    graph: MikadoGraph
    ensure_clean: Callable[[int], None]
    remove: Callable[[int, Path], None]


def fail_bounded_attempt(
    runtime: AttemptRuntime, node: MikadoNode, context: NodeExecutionContext, detail: str | None
) -> None:
    assert node.run_id is not None and context.attempt_id is not None
    attempt = runtime.graph.task_attempts.get(context.attempt_id)
    if attempt is None:
        raise ValueError(f"task attempt missing for node {node.id}")
    preserved = None
    if attempt.max_node_attempts == 1 or attempt.start_state == "executed":
        runtime.ensure_clean(node.id)
        preserved = discard_worktree(runtime.remove, node, "failed")
    result = RunResult(
        status="failed",
        exit_code=None,
        timed_out=False,
        ended_at=datetime.now(UTC).isoformat(),
        detail=preserved or detail or "node failed",
    )
    disposition = runtime.graph.task_attempts.fail(
        FinalizeTaskFailure(
            node.id, context.attempt_id, node.run_id, context.worker_run_id, result
        )
    )
    if disposition.status == "lost_fence":
        raise ValueError(f"task attempt fence lost for node {node.id}")


def supersede_review_round(
    finalize: Callable[[str | None, RunResult], None],
    node: MikadoNode,
    context: NodeExecutionContext | None,
) -> tuple[str | None, str]:
    owner_fence = context.owner_fence if context else None
    prior_worker_run_id = context.worker_run_id if context else None
    old_worker_run_id = prior_worker_run_id or node.run_id
    if not old_worker_run_id:
        raise ValueError(f"node {node.id} has no run fence/worker run id for review redispatch")
    # Finalize only the recorded worker run. The node fence may be a live parent run.
    review_round = context.review_round if context else 0
    finalize(
        prior_worker_run_id,
        RunResult(
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at=datetime.now(UTC).isoformat(),
            detail=f"superseded by review round {review_round} redispatch",
        ),
    )
    return owner_fence, old_worker_run_id


def bind_before_start(
    graph: MikadoGraph, run_id: str, before_start: Callable[[str], None] | None
) -> None:
    if before_start is None:
        return
    try:
        before_start(run_id)
    except Exception:
        graph.runs.finish(
            run_id,
            RunResult(
                status="failed",
                exit_code=None,
                timed_out=False,
                ended_at=datetime.now(UTC).isoformat(),
                error="dispatch aborted",
            ),
        )
        raise
