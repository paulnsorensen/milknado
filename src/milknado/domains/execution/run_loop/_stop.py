"""Run-loop stop controls shared by CLI and detached supervision."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common import LoopPort
    from milknado.domains.execution.executor import Executor


class StopControlMixin(ABC):
    _active: dict[str, int]
    _executor: Executor
    _loop: LoopPort
    _scheduling_lock: Lock
    _scheduling_stopped: bool

    @abstractmethod
    def _publish_state(self) -> None: ...

    def force_stop(self, run_id: str, timeout: float = 10.0) -> bool:
        stopped = self._executor.force_stop_run(run_id, timeout)
        self._publish_state()
        return stopped

    def force_stop_active(self, deadline: float) -> bool:
        self._scheduling_stopped = True
        stopped = self._loop.stop_active_workers(deadline)
        for run_id in tuple(self._active):
            remaining = max(0.0, deadline - time.monotonic())
            stopped = self._executor.force_stop_run(run_id, remaining) and stopped
        return stopped

    def admit_stop_scheduling(self) -> None:
        """Close scheduling admission before a graceful control is queued."""
        with self._scheduling_lock:
            self._scheduling_stopped = True

    def stop_scheduling(self) -> None:
        """Prevent redispatch and request a graceful stop for active runs."""
        self.admit_stop_scheduling()
        for run_id in self._active:
            self._loop.request_stop_run(run_id)
        self._publish_state()
