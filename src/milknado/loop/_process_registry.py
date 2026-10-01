"""Independent admission and shutdown registry for protected loop launches."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import final

_log = logging.getLogger(__name__)


@final
class LaunchTicket:
    def __init__(self, registry: WorkerRegistry, graph_run_id: str | None) -> None:
        self._registry = registry
        self.graph_run_id = graph_run_id
        self._gate_close: Callable[[], None] | None = None
        self._shutdown: Callable[[float], bool] | None = None

    @property
    def cancelled(self) -> bool:
        with self._registry.lock:
            return self._registry.requested()

    def bind_pending(self, close_gate: Callable[[], None]) -> bool:
        with self._registry.lock:
            self._gate_close = close_gate
            cancelled = self._registry.requested()
        if cancelled:
            close_gate()
        return not cancelled

    def activate(
        self, release_gate: Callable[[], None], shutdown: Callable[[float], bool]
    ) -> bool:
        with self._registry.lock:
            if self._registry.requested():
                return False
            self._shutdown = shutdown
            release_gate()
            return True

    def stop(self, deadline: float) -> bool:
        with self._registry.lock:
            close_gate, shutdown = self._gate_close, self._shutdown
        if shutdown is not None:
            return shutdown(deadline)
        if close_gate is not None:
            close_gate()
        return False

    def close(self) -> None:
        with self._registry.lock:
            self._registry.tickets.discard(self)


@final
class WorkerRegistry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tickets: set[LaunchTicket] = set()
        self._stopping = False
        self._deadline: float | None = None
        self._shutdown_intent: Callable[[], bool] = lambda: False

    def bind_shutdown_intent(self, requested: Callable[[], bool]) -> None:
        with self.lock:
            self._shutdown_intent = requested

    def requested(self) -> bool:
        return self._stopping or self._shutdown_intent()

    def reserve(self, graph_run_id: str | None = None) -> LaunchTicket:
        with self.lock:
            if self.requested():
                raise RuntimeError("worker admission is closed")
            ticket = LaunchTicket(self, graph_run_id)
            self.tickets.add(ticket)
            return ticket

    def stop_all(self, deadline: float) -> bool:
        with self.lock:
            self._stopping = True
            self._deadline = deadline if self._deadline is None else min(self._deadline, deadline)
            active = tuple(self.tickets)
            effective_deadline = self._deadline
        return self._stop_tickets(active, effective_deadline)

    def stop_run_workers(self, graph_run_id: str, deadline: float) -> bool:
        with self.lock:
            active = tuple(
                ticket for ticket in self.tickets if ticket.graph_run_id == graph_run_id
            )
        return self._stop_tickets(active, deadline)

    @staticmethod
    def _stop_tickets(active: tuple[LaunchTicket, ...], deadline: float) -> bool:
        results = [False] * len(active)

        def stop_one(index: int, ticket: LaunchTicket) -> None:
            try:
                results[index] = ticket.stop(deadline)
            except Exception:  # noqa: BLE001
                _log.exception("worker shutdown failed")

        threads = [
            threading.Thread(target=stop_one, args=(index, ticket), daemon=True)
            for index, ticket in enumerate(active)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        return all(results) and all(not thread.is_alive() for thread in threads)
