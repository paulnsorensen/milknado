from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from threading import Lock
from typing import final

from milknado.domains.common import ProgressEvent
from milknado.domains.execution.run_loop.state import TerminalRunState


@dataclass(frozen=True, slots=True)
class ActiveRun:
    run_id: str
    node_id: int
    dispatched_at: float
    prior_attempts: int
    progress: ProgressEvent | None


@dataclass(frozen=True, slots=True)
class SchedulerView:
    active: tuple[ActiveRun, ...]
    terminal_runs: tuple[TerminalRunState, ...]
    completion_durations: tuple[float, ...]
    stopped_nodes: frozenset[int]
    completed: int
    failed: int
    stopped: int
    failure_triggered: bool
    capacity_deferred: bool
    scheduling_stopped: bool


@dataclass(frozen=True, slots=True)
class FinishedRun:
    node_id: int
    duration: float


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    available: int
    stopped_nodes: frozenset[int]


@dataclass(frozen=True, slots=True)
class DispatchFailure:
    stop_batch: bool


@final
class Scheduler:
    """Own run admission, lifecycle history, and scheduling decisions."""

    def __init__(self, eta_sample_size: int) -> None:
        self._lock = Lock()
        self._active: dict[str, int] = {}
        self._dispatched_at: dict[str, float] = {}
        self._attempts: dict[int, int] = {}
        self._progress_by_run: dict[str, ProgressEvent] = {}
        self._terminal_runs: deque[TerminalRunState] = deque(maxlen=20)
        self._completion_durations: deque[float] = deque(maxlen=eta_sample_size)
        self._stopped_nodes: set[int] = set()
        self._completed = 0
        self._failed = 0
        self._stopped = 0
        self._failure_triggered = False
        self._capacity_deferred = False
        self._deferred_retry_at = 0.0
        self._scheduling_stopped = False

    def view(self) -> SchedulerView:
        with self._lock:
            return SchedulerView(
                active=tuple(
                    ActiveRun(
                        run_id,
                        node_id,
                        self._dispatched_at[run_id],
                        self._attempts.get(node_id, 0),
                        self._progress_by_run.get(run_id),
                    )
                    for run_id, node_id in self._active.items()
                ),
                terminal_runs=tuple(self._terminal_runs),
                completion_durations=tuple(self._completion_durations),
                stopped_nodes=frozenset(self._stopped_nodes),
                completed=self._completed,
                failed=self._failed,
                stopped=self._stopped,
                failure_triggered=self._failure_triggered,
                capacity_deferred=self._capacity_deferred,
                scheduling_stopped=self._scheduling_stopped,
            )

    def reset_run(self) -> None:
        with self._lock:
            self._stopped_nodes.clear()
            self._terminal_runs.clear()
            self._completed = self._failed = self._stopped = 0

    def admit_run(self, run_id: str, node_id: int, now: float) -> None:
        with self._lock:
            self._active[run_id] = node_id
            self._dispatched_at[run_id] = now

    def record_progress(self, event: ProgressEvent) -> None:
        with self._lock:
            self._progress_by_run[event.run_id] = event

    def finish_run(self, run_id: str, terminal: TerminalRunState, now: float) -> FinishedRun:
        with self._lock:
            node_id = self._active.pop(run_id)
            _ = self._progress_by_run.pop(run_id, None)
            duration = now - self._dispatched_at.pop(run_id, now)
            self._terminal_runs.append(replace(terminal, duration_seconds=duration))
            return FinishedRun(node_id, duration)

    def abandon_run(self, run_id: str) -> int | None:
        with self._lock:
            node_id = self._active.pop(run_id, None)
            _ = self._dispatched_at.pop(run_id, None)
            _ = self._progress_by_run.pop(run_id, None)
            return node_id

    def record_completion(self, duration: float) -> None:
        with self._lock:
            self._completion_durations.append(duration)

    def record_failure(self, node_id: int, strict: bool) -> None:
        with self._lock:
            self._attempts[node_id] = self._attempts.get(node_id, 0) + 1
            if strict:
                self._failure_triggered = True

    def trigger_failure(self) -> None:
        with self._lock:
            self._failure_triggered = True

    def record_stop(self, node_id: int) -> None:
        with self._lock:
            self._stopped_nodes.add(node_id)
            self._stopped += 1

    def count_outcomes(self, completed: int = 0, failed: int = 0) -> None:
        with self._lock:
            self._completed += completed
            self._failed += failed

    def plan_dispatch(self, concurrency_limit: int, strict: bool) -> DispatchPlan:
        with self._lock:
            self._capacity_deferred = False
            available = (
                0
                if (self._scheduling_stopped or (strict and self._failure_triggered))
                else max(0, concurrency_limit - len(self._active))
            )
            return DispatchPlan(available, frozenset(self._stopped_nodes))

    def dispatch_failed(self, strict: bool) -> DispatchFailure:
        with self._lock:
            if strict:
                self._failure_triggered = True
            return DispatchFailure(stop_batch=strict)

    def defer_capacity(self) -> None:
        with self._lock:
            self._capacity_deferred = True

    def retry_deferred(self, now: float, interval: float) -> bool:
        with self._lock:
            if not self._capacity_deferred or now < self._deferred_retry_at:
                return False
            self._deferred_retry_at = now + interval
            return True

    def close_admission(self) -> None:
        with self._lock:
            self._scheduling_stopped = True
