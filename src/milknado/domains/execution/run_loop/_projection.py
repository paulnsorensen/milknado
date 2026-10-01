from __future__ import annotations

from milknado.domains.execution.run_loop.state import (
    ActiveRunFacts,
    ActiveRunState,
    ProjectionFacts,
    RunActionState,
    RunLoopState,
)
from milknado.loop import RunStatus


def _action_reasons(run: ActiveRunFacts) -> RunActionState:
    terminal_reason = {
        RunStatus.COMPLETED: "run has completed",
        RunStatus.FAILED: "run has failed",
        RunStatus.STOPPED: "run has stopped",
    }.get(run.status)
    cancel_reason = guidance_reason = force_stop_reason = terminal_reason
    if terminal_reason is None and run.stop_requested:
        cancel_reason = "stop already requested"
        guidance_reason = "run is stopping"
    if terminal_reason is None and run.force_stop_requested:
        force_stop_reason = "force stop already requested"
    return RunActionState(cancel_reason, guidance_reason, force_stop_reason)


def _active_state(
    run: ActiveRunFacts, now: float, average_duration: float | None, facts: ProjectionFacts
) -> ActiveRunState:
    elapsed = now - run.dispatched_at if run.dispatched_at is not None else 0.0
    progress = run.progress_message or (
        f"{run.progress_work}/{run.progress_total}" if run.progress_work is not None else None
    )
    progress_pct = (
        run.progress_work / run.progress_total * 100
        if (
            run.progress_work is not None
            and run.progress_total is not None
            and run.progress_total > 0
        )
        else None
    )
    return ActiveRunState(
        run_id=run.run_id,
        node_id=run.node_id,
        description=run.description,
        status=run.status,
        progress=progress,
        stop_requested=run.stop_requested,
        actions=_action_reasons(run),
        output=run.output[-30:],
        pending_guidance=run.pending_guidance,
        elapsed_seconds=elapsed,
        progress_pct=progress_pct,
        eta_seconds=max(0.0, average_duration - elapsed) if average_duration is not None else None,
        attempt=run.prior_attempts + 1,
        max_attempts=facts.max_attempts,
        stalled=progress_pct is None and elapsed >= facts.stall_threshold_seconds,
        session=run.session,
    )


def project_state(facts: ProjectionFacts, now: float) -> RunLoopState:
    durations = facts.completion_durations
    average = sum(durations) / len(durations) if len(durations) >= 3 else None
    return RunLoopState(
        goal=facts.goal,
        active_runs=tuple(_active_state(run, now, average, facts) for run in facts.active_runs),
        terminal_runs=facts.terminal_runs,
        completed=facts.completed,
        failed=facts.failed,
        stopped=facts.stopped,
        available=facts.available,
        event_lines=facts.event_lines,
        execution_agent=facts.execution_agent,
        log_path=facts.log_path,
    )
