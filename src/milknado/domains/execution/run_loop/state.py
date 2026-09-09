from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from milknado.domains.common import ProgressEvent
from milknado.loop import RunStatus

_US_PREFIX_RE = re.compile(r"^US-\d+:\s*")


def average_duration(durations: Sequence[float]) -> float | None:
    values = list(durations)
    return sum(values) / len(values) if len(values) >= 3 else None


def action_reasons(
    status: RunStatus, stop_requested: bool, force_stop_requested: bool
) -> tuple[str | None, str | None, str | None]:
    terminal_reason = {
        RunStatus.COMPLETED: "run has completed",
        RunStatus.FAILED: "run has failed",
        RunStatus.STOPPED: "run has stopped",
    }.get(status)
    cancel_reason = guidance_reason = force_stop_reason = terminal_reason
    if terminal_reason is None and stop_requested:
        cancel_reason = "stop already requested"
        guidance_reason = "run is stopping"
    if terminal_reason is None and force_stop_requested:
        force_stop_reason = "force stop already requested"
    return cancel_reason, guidance_reason, force_stop_reason


def progress_state(event: ProgressEvent | None) -> tuple[str | None, float | None]:
    if event is None:
        return None, None
    progress = event.message or f"{event.work}/{event.total}"
    progress_pct = event.work / event.total * 100 if event.total > 0 else None
    return progress, progress_pct


def summarize_description(description: str, max_chars: int = 80) -> str:
    text = description.split("\n", 1)[0]
    text = _US_PREFIX_RE.sub("", text)
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


@dataclass(frozen=True, slots=True)
class RunActionState:
    cancel_reason: str | None = None
    guidance_reason: str | None = None
    force_stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ActiveRunState:
    run_id: str
    node_id: int
    description: str
    status: RunStatus
    progress: str | None
    stop_requested: bool
    actions: RunActionState
    output: tuple[str, ...]
    pending_guidance: tuple[str, ...]
    elapsed_seconds: float
    progress_pct: float | None
    eta_seconds: float | None
    attempt: int
    max_attempts: int
    stalled: bool


@dataclass(frozen=True, slots=True)
class TerminalRunState:
    run_id: str
    node_id: int
    description: str
    status: RunStatus
    output: tuple[str, ...]
    pending_guidance: tuple[str, ...]
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class RunLoopState:
    goal: str
    active_runs: tuple[ActiveRunState, ...]
    terminal_runs: tuple[TerminalRunState, ...]
    completed: int
    failed: int
    stopped: int
    available: int
    event_lines: tuple[str, ...]
    execution_agent: str = "(unknown)"
    log_path: str | None = None
