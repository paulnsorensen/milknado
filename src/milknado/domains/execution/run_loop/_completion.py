from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from milknado.domains.common import TerminalRunOutcome
from milknado.domains.execution._models import RebaseConflict
from milknado.domains.execution.run_loop._logging import ts
from milknado.domains.execution.run_loop._scheduler import Scheduler
from milknado.domains.execution.run_loop.state import TerminalRunState
from milknado.loop import RunStatus

if TYPE_CHECKING:
    from milknado.domains.common.protocols import LoopPort
    from milknado.domains.execution.executor import Executor
    from milknado.domains.execution.run_loop.input import InputState
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger("milknado")
TerminalRunStatus = Literal["completed", "stopped", "failed"]


@dataclass(frozen=True, slots=True)
class CompletionContext:
    scheduler: Scheduler
    graph: MikadoGraph
    executor: Executor
    loop: LoopPort
    input_state: InputState
    logs: deque[str]
    strict: bool


def handle_completion(
    context: CompletionContext,
    run_id: str,
    outcome: TerminalRunStatus | TerminalRunOutcome,
    feature_branch: str,
) -> tuple[int, int, list[RebaseConflict]]:
    status = outcome.status if isinstance(outcome, TerminalRunOutcome) else outcome
    scheduler = context.scheduler
    graph = context.graph
    executor = context.executor
    loop_adapter = context.loop
    logs = context.logs
    input_state = context.input_state
    strict = context.strict

    active = next(run for run in scheduler.view().active if run.run_id == run_id)
    node_id = active.node_id
    node = graph.get_node(node_id)
    description = node.description if node else str(node_id)
    terminal = TerminalRunState(
        run_id=run_id,
        node_id=node_id,
        description=description,
        status=RunStatus(status),
        output=tuple(loop_adapter.get_run_output_tail(run_id, 30)),
        pending_guidance=tuple(loop_adapter.get_run_guidance(run_id)),
        duration_seconds=0.0,
        session=loop_adapter.get_run_session(run_id),
    )
    finished = scheduler.finish_run(run_id, terminal, time.monotonic())
    duration = finished.duration
    if input_state.overlay_state == run_id:
        input_state.overlay_state = None

    if status == "completed":
        result = executor.complete(node_id, feature_branch)
        if result.review_notification_failed:
            logs.append(f"[{ts()}] ! node {node_id} review notification failed")
        if result.review_audit_failed:
            logs.append(f"[{ts()}] ! node {node_id} review audit failed")
        if result.redispatch is not None:
            redispatch = result.redispatch
            scheduler.admit_run(redispatch.run_id, node_id, time.monotonic())
            _logger.info("node_review_redispatch node_id=%d run_id=%s", node_id, redispatch.run_id)
            logs.append(f"[{ts()}] ↻ node {node_id} review round")
            return 0, 0, []
        scheduler.record_completion(duration)
        if result.blocked:
            _logger.warning("node_review_blocked node_id=%d", node_id)
            logs.append(f"[{ts()}] ■ node {node_id} review blocked")
            scheduler.record_failure(node_id, strict)
            return 0, 1, []
        if result.rebase_conflict:
            conflict = result.rebase_conflict
            _logger.warning(
                "node_conflict node_id=%d files=%s", node_id, list(conflict.conflicting_files)
            )
            logs.append(f"[{ts()}] ✗ node {node_id} conflict")
            scheduler.record_failure(node_id, strict)
            return 0, 1, [conflict]
        if not result.rebased:
            _logger.warning("node_completion_unrebased node_id=%d", node_id)
            logs.append(f"[{ts()}] ✗ node {node_id} did not complete")
            scheduler.record_failure(node_id, strict)
            return 0, 1, []
        _logger.info("node_completed node_id=%d duration=%.1fs", node_id, duration)
        logs.append(f"[{ts()}] ✓ node {node_id} in {int(duration)}s")
        return 1, 0, []

    if status == "stopped":
        executor.cancel(node_id)
        scheduler.record_stop(node_id)
        _logger.info("node_stopped node_id=%d run_id=%s duration=%.1fs", node_id, run_id, duration)
        logs.append(f"[{ts()}] ■ node {node_id} stopped")
        return 0, 0, []

    detail = loop_adapter.get_run_failure_detail(run_id)
    executor.fail(node_id, detail=detail)
    if detail:
        _logger.warning("node_failed node_id=%d detail=%s", node_id, detail)
    else:
        _logger.warning("node_failed node_id=%d", node_id)
    logs.append(f"[{ts()}] ✗ node {node_id} failed")
    scheduler.record_failure(node_id, strict)
    return 0, 1, []
