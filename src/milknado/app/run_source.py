"""Shared immutable snapshot contracts for run and watch surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from milknado.domains.common import SessionView
from milknado.domains.graph import GraphSnapshot, NodeDetailResponse


class ExecutionRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class RunActionAvailability:
    cancel_reason: str | None = None
    guidance_reason: str | None = None
    force_stop_reason: str | None = None

    @property
    def can_cancel(self) -> bool:
        return self.cancel_reason is None

    @property
    def can_queue_guidance(self) -> bool:
        return self.guidance_reason is None

    @property
    def can_force_stop(self) -> bool:
        return self.force_stop_reason is None


@dataclass(frozen=True, slots=True)
class ActiveRunSnapshot:
    run_id: str
    node_id: int
    description: str
    status: ExecutionRunStatus
    progress: str | None
    stop_requested: bool
    actions: RunActionAvailability
    output: tuple[str, ...]
    pending_guidance: tuple[str, ...] | None
    elapsed_seconds: float
    progress_pct: float | None
    eta_seconds: float | None
    attempt: int | None
    max_attempts: int | None
    stalled: bool
    session: SessionView = SessionView()


@dataclass(frozen=True, slots=True)
class TerminalRunSnapshot:
    run_id: str
    node_id: int
    description: str
    status: ExecutionRunStatus
    output: tuple[str, ...]
    pending_guidance: tuple[str, ...] | None
    duration_seconds: float
    session: SessionView = SessionView()


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    goal: str
    active_runs: tuple[ActiveRunSnapshot, ...]
    terminal_runs: tuple[TerminalRunSnapshot, ...]
    completed: int
    failed: int
    stopped: int
    available: int
    event_lines: tuple[str, ...]
    listener_errors: tuple[str, ...] = ()
    graph: GraphSnapshot | None = None
    node: NodeDetailResponse | None = None


@dataclass(frozen=True, slots=True)
class NodeSnapshotRequest:
    node_id: int
    request_generation: int
    page: int = 0
    limit: int = 50


class ExecutionSnapshotSource(Protocol):
    def snapshot(self) -> ExecutionSnapshot: ...

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]: ...
    def node_snapshot(  # noqa: V105 - shared source contract consumed by run and watch clients
        self, request: NodeSnapshotRequest
    ) -> NodeDetailResponse: ...
