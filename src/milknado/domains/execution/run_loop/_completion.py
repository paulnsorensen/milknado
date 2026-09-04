from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from milknado.domains.common import TerminalRunOutcome
from milknado.domains.execution.executor import RebaseConflict
from milknado.domains.execution.run_loop._logging import ts
from milknado.domains.execution.run_loop.display import _summarize_description
from milknado.domains.execution.run_loop.state import TerminalRunState
from milknado.loop import RunStatus

if TYPE_CHECKING:
    from rich.live import Live

    from milknado.domains.execution.run_loop import RunLoop

_logger = logging.getLogger("milknado")


def handle_completion(
    loop: RunLoop,
    run_id: str,
    outcome: TerminalRunOutcome,
    feature_branch: str,
    live: Live | None,
) -> tuple[int, int, list[RebaseConflict]]:
    completed = failed = 0
    conflicts: list[RebaseConflict] = []
    active = loop._active
    progress_by_run = loop._progress_by_run
    input_state = loop._input
    graph = loop._graph
    dispatched_at = loop._dispatched_at
    logs = loop._logs

    node_id = active.pop(run_id)
    _ = progress_by_run.pop(run_id, None)
    if input_state.overlay_state == run_id:
        input_state.overlay_state = None
    node = graph.get_node(node_id)
    desc = _summarize_description(node.description) if node else str(node_id)
    start = dispatched_at.pop(run_id, time.monotonic())
    duration = time.monotonic() - start

    if outcome == "completed":
        executor = loop._executor
        completion_durations = loop._completion_durations
        result = executor.complete(node_id, feature_branch)
        if result.review_notification_failed:
            # Orthogonal to the outcome below: the review ran, but its verdict could
            # not be delivered. Surface it before any early return so the operator
            # sees a review whose result may never have reached the worker.
            if live is not None:
                live.console.print(
                    f"[yellow]![/yellow] [{node_id}] {desc} — review notification failed"
                )
            logs.append(f"[{ts()}] ! node {node_id} review notification failed")
        if result.redispatch is not None:
            redispatch = result.redispatch
            active[redispatch.run_id] = node_id
            dispatched_at[redispatch.run_id] = time.monotonic()
            if live is not None:
                live.console.print(
                    f"[yellow]↻[/yellow] [{node_id}] {desc} — "
                    + "adversarial review requested another round"
                )
            _logger.info(
                "node_review_redispatch node_id=%d run_id=%s",
                node_id,
                redispatch.run_id,
            )
            logs.append(f"[{ts()}] ↻ node {node_id} review round")
            return completed, failed, conflicts
        completion_durations.append(duration)
        attempts = loop._attempts
        strict = loop._strict
        if result.blocked:
            if live is not None:
                live.console.print(f"[red]■[/red] [{node_id}] {desc} — review blocked")
            _logger.warning("node_review_blocked node_id=%d", node_id)
            logs.append(f"[{ts()}] ■ node {node_id} review blocked")
            attempts[node_id] = attempts.get(node_id, 0) + 1
            if strict:
                loop._failure_triggered = True
            failed += 1
        elif result.rebase_conflict:
            conflicts.append(result.rebase_conflict)
            files = ", ".join(result.rebase_conflict.conflicting_files)
            if live is not None:
                live.console.print(f"[red]✗[/red] [{node_id}] {desc} — conflict: {files}")
            _logger.warning(
                "node_conflict node_id=%d files=%s",
                node_id,
                list(result.rebase_conflict.conflicting_files),
            )
            logs.append(f"[{ts()}] ✗ node {node_id} conflict")
            attempts[node_id] = attempts.get(node_id, 0) + 1
            if strict:
                loop._failure_triggered = True
            failed += 1
        else:
            if live is not None:
                live.console.print(f"[green]✓[/green] [{node_id}] {desc}")
            _logger.info("node_completed node_id=%d duration=%.1fs", node_id, duration)
            logs.append(f"[{ts()}] ✓ node {node_id} in {int(duration)}s")
            completed += 1
    elif outcome == "stopped":
        ralph = loop._ralph
        executor = loop._executor
        stopped_nodes = loop._stopped_nodes
        stopped = loop._stopped
        terminal_runs = loop._terminal_runs
        output = tuple(ralph.get_run_output_tail(run_id, 30))
        pending_guidance = tuple(ralph.get_run_guidance(run_id))
        executor.cancel(node_id)
        stopped_nodes.add(node_id)
        stopped += 1
        loop._stopped = stopped
        terminal_runs.append(
            TerminalRunState(
                run_id=run_id,
                node_id=node_id,
                description=desc,
                status=RunStatus.STOPPED,
                output=output,
                pending_guidance=pending_guidance,
                duration_seconds=duration,
            )
        )
        if live is not None:
            live.console.print(f"[yellow]■[/yellow] [{node_id}] {desc} — stopped")
        _logger.info(
            "node_stopped node_id=%d run_id=%s duration=%.1fs",
            node_id,
            run_id,
            duration,
        )
        logs.append(f"[{ts()}] ■ node {node_id} stopped")
    else:
        ralph = loop._ralph
        executor = loop._executor
        attempts = loop._attempts
        strict = loop._strict
        detail = ralph.get_run_failure_detail(run_id)
        executor.fail(node_id, detail=detail)
        if live is not None:
            live.console.print(f"[red]✗[/red] [{node_id}] {desc}")
        if detail:
            _logger.warning("node_failed node_id=%d detail=%s", node_id, detail)
        else:
            _logger.warning("node_failed node_id=%d", node_id)
        logs.append(f"[{ts()}] ✗ node {node_id} failed")
        attempts[node_id] = attempts.get(node_id, 0) + 1
        if strict:
            loop._failure_triggered = True
        failed += 1

    return completed, failed, conflicts
