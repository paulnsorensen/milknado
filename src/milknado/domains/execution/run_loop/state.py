from __future__ import annotations

from dataclasses import dataclass

from milknado.loop import RunStatus


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


@dataclass(frozen=True, slots=True)
class TerminalRunState:
    run_id: str
    node_id: int
    description: str
    status: RunStatus
    output: tuple[str, ...]
    pending_guidance: tuple[str, ...]


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
