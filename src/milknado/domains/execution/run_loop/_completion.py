from __future__ import annotations

import logging
import time

from milknado.domains.common import TerminalRunOutcome
from milknado.domains.execution.executor import RebaseConflict
from milknado.domains.execution.run_loop._logging import ts
from milknado.domains.execution.run_loop._protocols import RunLoopState
from milknado.domains.execution.run_loop.state import TerminalRunState
from milknado.loop import RunStatus

_logger = logging.getLogger("milknado")


def handle_completion(
    loop: RunLoopState,
    run_id: str,
    outcome: TerminalRunOutcome,
    feature_branch: str,
) -> tuple[int, int, list[RebaseConflict]]:
    completed = failed = 0
    conflicts: list[RebaseConflict] = []
    active = loop._active  # pyright: ignore[reportPrivateUsage]
    progress_by_run = loop._progress_by_run  # pyright: ignore[reportPrivateUsage]
    input_state = loop._input  # pyright: ignore[reportPrivateUsage]
    graph = loop._graph  # pyright: ignore[reportPrivateUsage]
    dispatched_at = loop._dispatched_at  # pyright: ignore[reportPrivateUsage]
    logs = loop._logs  # pyright: ignore[reportPrivateUsage]

    node_id = active.pop(run_id)
    _ = progress_by_run.pop(run_id, None)
    if input_state.overlay_state == run_id:
        input_state.overlay_state = None
    node = graph.get_node(node_id)
    desc = node.description if node else str(node_id)
    start = dispatched_at.pop(run_id, time.monotonic())
    duration = time.monotonic() - start
    ralph = loop._ralph  # pyright: ignore[reportPrivateUsage]
    loop._terminal_runs.append(  # pyright: ignore[reportPrivateUsage]
        TerminalRunState(
            run_id=run_id,
            node_id=node_id,
            description=desc,
            status=RunStatus(outcome),
            output=tuple(ralph.get_run_output_tail(run_id, 30)),
            pending_guidance=tuple(ralph.get_run_guidance(run_id)),
            duration_seconds=duration,
            session=ralph.get_run_session(run_id),
        )
    )

    if outcome == "completed":
        executor = loop._executor  # pyright: ignore[reportPrivateUsage]
        completion_durations = loop._completion_durations  # pyright: ignore[reportPrivateUsage]
        result = executor.complete(node_id, feature_branch)
        if result.review_notification_failed:
            # Orthogonal to the outcome below: the review ran, but its verdict could
            # not be delivered. Surface it before any early return so the operator
            # sees a review whose result may never have reached the worker.
            logs.append(f"[{ts()}] ! node {node_id} review notification failed")
        if result.review_audit_failed:
            logs.append(f"[{ts()}] ! node {node_id} review audit failed")
        if result.redispatch is not None:
            redispatch = result.redispatch
            active[redispatch.run_id] = node_id
            dispatched_at[redispatch.run_id] = time.monotonic()
            _logger.info(
                "node_review_redispatch node_id=%d run_id=%s",
                node_id,
                redispatch.run_id,
            )
            logs.append(f"[{ts()}] ↻ node {node_id} review round")
            return completed, failed, conflicts
        completion_durations.append(duration)
        attempts = loop._attempts  # pyright: ignore[reportPrivateUsage]
        strict = loop._strict  # pyright: ignore[reportPrivateUsage]
        if result.blocked:
            _logger.warning("node_review_blocked node_id=%d", node_id)
            logs.append(f"[{ts()}] ■ node {node_id} review blocked")
            attempts[node_id] = attempts.get(node_id, 0) + 1
            if strict:
                loop._failure_triggered = True  # pyright: ignore[reportPrivateUsage]
            failed += 1
        elif result.rebase_conflict:
            conflicts.append(result.rebase_conflict)
            _logger.warning(
                "node_conflict node_id=%d files=%s",
                node_id,
                list(result.rebase_conflict.conflicting_files),
            )
            logs.append(f"[{ts()}] ✗ node {node_id} conflict")
            attempts[node_id] = attempts.get(node_id, 0) + 1
            if strict:
                loop._failure_triggered = True  # pyright: ignore[reportPrivateUsage]
            failed += 1
        else:
            _logger.info("node_completed node_id=%d duration=%.1fs", node_id, duration)
            logs.append(f"[{ts()}] ✓ node {node_id} in {int(duration)}s")
            completed += 1
    elif outcome == "stopped":
        executor = loop._executor  # pyright: ignore[reportPrivateUsage]
        stopped_nodes = loop._stopped_nodes  # pyright: ignore[reportPrivateUsage]
        stopped = loop._stopped  # pyright: ignore[reportPrivateUsage]
        executor.cancel(node_id)
        stopped_nodes.add(node_id)
        stopped += 1
        loop._stopped = stopped  # pyright: ignore[reportPrivateUsage]
        _logger.info(
            "node_stopped node_id=%d run_id=%s duration=%.1fs",
            node_id,
            run_id,
            duration,
        )
        logs.append(f"[{ts()}] ■ node {node_id} stopped")
    else:
        executor = loop._executor  # pyright: ignore[reportPrivateUsage]
        attempts = loop._attempts  # pyright: ignore[reportPrivateUsage]
        strict = loop._strict  # pyright: ignore[reportPrivateUsage]
        detail = ralph.get_run_failure_detail(run_id)
        executor.fail(node_id, detail=detail)
        if detail:
            _logger.warning("node_failed node_id=%d detail=%s", node_id, detail)
        else:
            _logger.warning("node_failed node_id=%d", node_id)
        logs.append(f"[{ts()}] ✗ node {node_id} failed")
        attempts[node_id] = attempts.get(node_id, 0) + 1
        if strict:
            loop._failure_triggered = True  # pyright: ignore[reportPrivateUsage]
        failed += 1

    return completed, failed, conflicts
