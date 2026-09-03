"""Thread-safe lifecycle management for Milknado loop runs."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from milknado.loop._events import EventEmitter, QueueEmitter
from milknado.loop._run_types import RunConfig, RunState, generate_run_id
from milknado.loop.engine import run_loop


@dataclass(slots=True)
class ManagedRun:
    """A run bundled with its background thread and event emitter."""

    config: RunConfig
    state: RunState
    emitter: EventEmitter
    thread: threading.Thread | None = None


class RunManager:
    """Create, start, stop, and inspect background loop runs."""

    def __init__(self) -> None:
        self._runs: dict[str, ManagedRun] = {}
        self._lock = threading.Lock()

    def _lookup(self, run_id: str) -> ManagedRun:
        """Look up a run by ID. The caller must hold ``_lock``."""
        try:
            return self._runs[run_id]
        except KeyError:
            raise KeyError(f"No run with ID '{run_id}'") from None

    def _require_run(self, run_id: str) -> ManagedRun:
        """Look up a run by ID under the registry lock."""
        with self._lock:
            return self._lookup(run_id)

    def create_run(
        self,
        config: RunConfig,
        emitter: EventEmitter | None = None,
        *,
        run_id: str | None = None,
    ) -> ManagedRun:
        """Create and register a run."""
        state = RunState(run_id=run_id or generate_run_id())
        managed = ManagedRun(
            config=config,
            state=state,
            emitter=emitter if emitter is not None else QueueEmitter(),
        )
        with self._lock:
            self._runs[state.run_id] = managed
        return managed

    def start_run(self, run_id: str) -> None:
        """Start the run loop in a daemon thread."""
        with self._lock:
            managed = self._lookup(run_id)
            if managed.thread is not None:
                raise RuntimeError(f"Run '{run_id}' has already been started")
            managed.thread = threading.Thread(
                target=run_loop,
                args=(managed.config, managed.state, managed.emitter),
                daemon=True,
                name=f"run-{run_id}",
            )
            managed.thread.start()

    def queue_guidance(self, run_id: str, text: str) -> bool:
        """Queue operator guidance for the run's next prompt."""
        return self._require_run(run_id).state.queue_guidance(text)

    def stop_run(self, run_id: str) -> None:
        """Signal the run to stop after the current iteration finishes."""
        self._require_run(run_id).state.request_stop()

    def force_stop_and_join(self, run_id: str, timeout: float | None = None) -> bool:
        """Force the active subprocess group down and report bounded completion."""
        managed = self._require_run(run_id)
        managed.state.request_force_stop()
        thread = managed.thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def stop_and_join(self, run_id: str, timeout: float | None = None) -> bool:
        """Request stop and wait until the run thread exits."""
        managed = self._require_run(run_id)
        managed.state.request_stop()
        thread = managed.thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def list_runs(self) -> list[ManagedRun]:
        """Return a snapshot of all registered runs."""
        with self._lock:
            return list(self._runs.values())

    def get_run(self, run_id: str) -> ManagedRun | None:
        """Look up a run by ID."""
        with self._lock:
            return self._runs.get(run_id)
