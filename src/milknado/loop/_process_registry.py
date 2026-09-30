"""Independent admission and shutdown registry for protected loop launches."""

from __future__ import annotations

import threading
from collections.abc import Callable


class LaunchTicket:
    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry
        self._gate_close: Callable[[], None] | None = None
        self._shutdown: Callable[[float], bool] | None = None

    @property
    def cancelled(self) -> bool:
        with self._registry._lock:
            return self._registry._requested()

    def bind_pending(self, close_gate: Callable[[], None]) -> bool:
        with self._registry._lock:
            self._gate_close = close_gate
            cancelled = self._registry._requested()
        if cancelled:
            close_gate()
        return not cancelled

    def activate(
        self, release_gate: Callable[[], None], shutdown: Callable[[float], bool]
    ) -> bool:
        with self._registry._lock:
            if self._registry._requested():
                return False
            self._shutdown = shutdown
            release_gate()
            return True

    def stop(self, deadline: float) -> bool:
        with self._registry._lock:
            close_gate, shutdown = self._gate_close, self._shutdown
        if shutdown is not None:
            return shutdown(deadline)
        if close_gate is not None:
            close_gate()
        return False

    def close(self) -> None:
        with self._registry._lock:
            self._registry._tickets.discard(self)


class WorkerRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tickets: set[LaunchTicket] = set()
        self._stopping = False
        self._deadline: float | None = None
        self._shutdown_intent: Callable[[], bool] = lambda: False

    def bind_shutdown_intent(self, requested: Callable[[], bool]) -> None:
        with self._lock:
            self._shutdown_intent = requested

    def _requested(self) -> bool:
        return self._stopping or self._shutdown_intent()

    def reserve(self) -> LaunchTicket:
        with self._lock:
            if self._requested():
                raise RuntimeError("worker admission is closed")
            ticket = LaunchTicket(self)
            self._tickets.add(ticket)
            return ticket

    def stop_all(self, deadline: float) -> bool:
        with self._lock:
            self._stopping = True
            self._deadline = deadline if self._deadline is None else min(self._deadline, deadline)
            active = tuple(self._tickets)
            effective_deadline = self._deadline
        results = [ticket.stop(effective_deadline) for ticket in active]
        return all(results)
