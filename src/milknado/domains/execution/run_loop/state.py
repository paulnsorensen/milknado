from __future__ import annotations

import re
from dataclasses import dataclass

from milknado.domains.common import SessionView
from milknado.loop import RunStatus

_US_PREFIX_RE = re.compile(r"^US-\d+:\s*")


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
    session: SessionView = SessionView()


@dataclass(frozen=True, slots=True)
class TerminalRunState:
    run_id: str
    node_id: int
    description: str
    status: RunStatus
    output: tuple[str, ...]
    pending_guidance: tuple[str, ...]
    duration_seconds: float
    session: SessionView = SessionView()


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


@dataclass(frozen=True, slots=True)
class ActiveRunFacts:
    run_id: str
    node_id: int
    description: str
    status: RunStatus
    stop_requested: bool
    force_stop_requested: bool
    output: tuple[str, ...]
    pending_guidance: tuple[str, ...]
    dispatched_at: float | None
    prior_attempts: int
    progress_message: str = ""
    progress_work: int | None = None
    progress_total: int | None = None
    session: SessionView = SessionView()


@dataclass(frozen=True, slots=True)
class ProjectionFacts:
    goal: str
    active_runs: tuple[ActiveRunFacts, ...]
    terminal_runs: tuple[TerminalRunState, ...]
    completed: int
    failed: int
    stopped: int
    available: int
    event_lines: tuple[str, ...]
    execution_agent: str
    log_path: str | None
    completion_durations: tuple[float, ...]
    stall_threshold_seconds: float
    max_attempts: int
