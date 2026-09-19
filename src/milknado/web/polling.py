"""Threaded polling source for read-only web clients."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from threading import Event, Lock, Thread, current_thread

from milknado.app.run_source import (
    ExecutionSnapshot,
    ExecutionSnapshotSource,
    NodeSnapshotRequest,
)
from milknado.domains.graph import NodeDetailResponse

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PolledSnapshotSource:
    """Cache snapshots while one background task polls a durable source."""

    source: ExecutionSnapshotSource
    interval: float = 1.0
    _snapshot: ExecutionSnapshot | None = field(default=None, init=False)
    _listeners: set[Callable[[ExecutionSnapshot], None]] = field(default_factory=set, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _stop: Event = field(default_factory=Event, init=False)
    _thread: Thread | None = field(default=None, init=False)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._snapshot = self.source.snapshot()
            self._stop.clear()
            self._thread = Thread(target=self._poll, name="milknado-web-poll", daemon=True)
            self._thread.start()

    def snapshot(self) -> ExecutionSnapshot:
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            self.start()
            with self._lock:
                snapshot = self._snapshot
        assert snapshot is not None
        return snapshot

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.add(listener)
            snapshot = self._snapshot
        if snapshot is not None:
            listener(snapshot)

        def unsubscribe() -> None:
            with self._lock:
                self._listeners.discard(listener)

        return unsubscribe

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        return self.source.node_snapshot(request)

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=max(self.interval * 2, 1.0))
        self._thread = None
        close = getattr(self.source, "close", None)
        if callable(close):
            _ = close()

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                snapshot = self.source.snapshot()
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                self._publish_error("poll", exc)
                continue
            with self._lock:
                self._snapshot = snapshot
                listeners = tuple(self._listeners)
            self._notify(listeners, snapshot)

    def _publish_error(self, operation: str, error: BaseException) -> None:
        message = f"Web snapshot {operation} failed: {type(error).__name__}: {error}"
        _logger.exception(message)
        with self._lock:
            if self._snapshot is None:
                return
            self._snapshot = replace(
                self._snapshot,
                listener_errors=(*self._snapshot.listener_errors, message),
            )
            snapshot = self._snapshot
            listeners = tuple(self._listeners)
        self._notify(listeners, snapshot)

    def _notify(
        self,
        listeners: tuple[Callable[[ExecutionSnapshot], None], ...],
        snapshot: ExecutionSnapshot,
    ) -> None:
        failures: list[str] = []
        for listener in listeners:
            try:
                listener(snapshot)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                name = getattr(listener, "__qualname__", type(listener).__qualname__)
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if failures:
            _logger.exception("Web snapshot listener failed: %s", "; ".join(failures))
            with self._lock:
                self._snapshot = replace(
                    self._snapshot or snapshot,
                    listener_errors=(
                        *(self._snapshot.listener_errors if self._snapshot else ()),
                        *failures,
                    ),
                )


__all__ = ["PolledSnapshotSource"]
