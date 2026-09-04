"""Pure string/style formatting for the execution TUI — no Textual imports."""

from __future__ import annotations

from rich.text import Text

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    TerminalRunSnapshot,
)

RunSnapshot = ActiveRunSnapshot | TerminalRunSnapshot

_STATUS_STYLES: dict[ExecutionRunStatus, str] = {
    ExecutionRunStatus.RUNNING: "cyan",
    ExecutionRunStatus.COMPLETED: "green",
    ExecutionRunStatus.FAILED: "red",
    ExecutionRunStatus.STOPPED: "yellow",
}


def format_duration(seconds: float) -> str:
    total = int(seconds)
    if total < 3600:
        minutes, secs = divmod(total, 60)
        return f"{minutes:02d}:{secs:02d}"
    hours, remainder = divmod(total, 3600)
    return f"{hours}h{remainder // 60:02d}m"


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "~?"
    return f"~{format_duration(seconds)}"


def format_progress(pct: float | None, stalled: bool) -> str:
    if pct is not None:
        clamped = max(0.0, min(100.0, pct))
        filled = max(0, min(10, int(clamped) // 10))
        bar = "█" * filled + "░" * (10 - filled)
        return f"{bar} {int(clamped)}%"
    if stalled:
        return "stalled"
    return "—"


def format_attempt(attempt: int | None, max_attempts: int | None) -> str:
    if attempt is None or max_attempts is None:
        return ""
    return "" if attempt == 1 else f"{attempt}/{max_attempts}"


def status_style(status: ExecutionRunStatus) -> str:
    return _STATUS_STYLES[status]


def subtitle_text(snapshot: ExecutionSnapshot) -> str:
    return (
        f"{len(snapshot.active_runs)} active · {snapshot.completed} completed · "
        f"{snapshot.failed} failed · {snapshot.stopped} stopped · "
        f"{snapshot.available} available"
    )


def summary_text(run: RunSnapshot | None) -> str:
    """Describe the selected run; the header already carries goal and totals."""
    if run is None:
        return "No runs."
    guidance = (
        "unavailable"
        if run.pending_guidance is None
        else ", ".join(run.pending_guidance) or "none"
    )
    if isinstance(run, ActiveRunSnapshot):
        stopping = " (stopping)" if run.stop_requested else ""
        attempt = (
            "unavailable"
            if run.attempt is None or run.max_attempts is None
            else f"{run.attempt}/{run.max_attempts}"
        )
        status = (
            f"{run.status.value}{stopping} · {run.progress or 'progress unavailable'} · "
            f"elapsed {format_duration(run.elapsed_seconds)} · "
            f"eta {format_eta(run.eta_seconds)} · attempt {attempt}"
        )
        guidance_label = "Pending guidance"
    else:
        status = f"{run.status.value} · ran {format_duration(run.duration_seconds)}"
        guidance_label = "Undelivered guidance"
    return (
        f"node {run.node_id} · {run.run_id}\n{run.description}\n"
        f"{status}\n{guidance_label}: {guidance}"
    )


def actions_text(run: RunSnapshot | None) -> str:
    if run is None:
        return "Actions\nNo run selected."
    if not isinstance(run, ActiveRunSnapshot):
        return "Actions\nRun has stopped; output retained for inspection."
    labeled = (
        ("Cancel", run.actions.cancel_reason),
        ("Force stop", run.actions.force_stop_reason),
        ("Guidance", run.actions.guidance_reason),
    )
    blocked = [f"{label}: {reason}" for label, reason in labeled if reason is not None]
    if not blocked:
        return "Actions\nAll actions available."
    return "Actions\n" + "\n".join(blocked)


def help_text(run: RunSnapshot | None, *, compact: bool, route: str, auto_follow: bool) -> str:
    actions = ["↑/↓ select", "h close help", "q quit"]
    if compact:
        actions.append("enter open" if route == "list" else "escape back")
    if isinstance(run, ActiveRunSnapshot):
        if run.actions.can_queue_guidance:
            actions.append("g queue guidance")
        if run.actions.can_cancel:
            actions.append("c cancel")
        if run.actions.can_force_stop:
            actions.append("f force stop")
    if not auto_follow:
        actions.append("r resume output")
    return "Help\n" + " · ".join(actions)


def output_border_title(*, auto_follow: bool) -> str:
    suffix = "following newest output" if auto_follow else "paused; press r to resume"
    return f"Output ({suffix})"


def output_body(run: RunSnapshot | None) -> str:
    return "\n".join(run.output) if run and run.output else "No output yet."


def events_text(event_lines: tuple[str, ...]) -> str:
    events = "\n".join(event_lines) or "No events yet."
    return f"Events\n{events}"


def run_row(run: RunSnapshot) -> tuple[str, str, Text, str, str]:
    """Build the five run-table cells; retries ride in the status cell, ETA in the summary."""
    if isinstance(run, ActiveRunSnapshot):
        progress = format_progress(run.progress_pct, run.stalled)
        elapsed = format_duration(run.elapsed_seconds)
        retry = format_attempt(run.attempt, run.max_attempts)
    else:
        progress = "—"
        elapsed = format_duration(run.duration_seconds)
        retry = ""
    label = f"{run.status.value} {retry}" if retry else run.status.value
    style = "bold red" if retry else status_style(run.status)
    return (str(run.node_id), run.description, Text(label, style=style), progress, elapsed)


def run_index(runs: tuple[RunSnapshot, ...], selected_run_id: str | None) -> int:
    ids = [run.run_id for run in runs]
    return ids.index(selected_run_id) if selected_run_id in ids else 0


def confirmation_text(action: str, run_id: str | None, active_run_count: int) -> str:
    if action == "force":
        return f"Force stop {run_id}? [y] confirm [n] cancel"
    label = "active run" if active_run_count == 1 else "active runs"
    return (
        f"Stop scheduling and gracefully stop {active_run_count} {label}? [y] confirm [n] cancel"
    )
