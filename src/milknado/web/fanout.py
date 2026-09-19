"""Thread-safe fan-out for execution snapshots."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import AsyncGenerator, Callable
from typing import cast

import msgspec

from milknado.app.run_source import ExecutionSnapshot, ExecutionSnapshotSource

_QUEUE_SIZE = 32
_logger = logging.getLogger(__name__)


class SnapshotFanout:
    """Bridge source callbacks into one bounded queue per connected stream."""

    def __init__(self, source: ExecutionSnapshotSource) -> None:
        self._source: ExecutionSnapshotSource = source
        self._clients: dict[
            asyncio.Queue[ExecutionSnapshot | None], asyncio.AbstractEventLoop
        ] = {}
        self._pending: dict[asyncio.Queue[ExecutionSnapshot | None], deque[ExecutionSnapshot]] = {}
        self._scheduled: set[asyncio.Queue[ExecutionSnapshot | None]] = set()
        self._unsubscribe: Callable[[], None] | None = None
        self._latest: ExecutionSnapshot | None = None
        self._subscribing: bool = False
        self._subscribing_clients: set[asyncio.Queue[ExecutionSnapshot | None]] = set()
        self._replayed_during_subscription: bool = False
        self._generation: int = 0
        self._lock: threading.RLock = threading.RLock()

    async def events(self) -> AsyncGenerator[dict[str, str], None]:
        queue: asyncio.Queue[ExecutionSnapshot | None] = asyncio.Queue(_QUEUE_SIZE)
        self._add(queue, asyncio.get_running_loop())
        try:
            while (snapshot := await queue.get()) is not None:
                self._schedule_drain(queue)
                yield {"event": "snapshot", "data": _encode(snapshot)}
        finally:
            self._remove(queue)

    def _add(
        self,
        queue: asyncio.Queue[ExecutionSnapshot | None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        generation = 0
        with self._lock:
            self._clients[queue] = loop
            self._pending[queue] = deque()
            if self._subscribing:
                self._subscribing_clients.add(queue)
                if self._replayed_during_subscription and self._latest is not None:
                    self._pending[queue].append(self._latest)
            elif self._unsubscribe is not None and self._latest is not None:
                self._pending[queue].append(self._latest)
            if self._unsubscribe is not None or self._subscribing:
                should_subscribe = False
            else:
                self._subscribing = True
                self._subscribing_clients = {queue}
                self._replayed_during_subscription = False
                self._generation += 1
                generation = self._generation
                should_subscribe = True
        if should_subscribe:
            self._subscribe(generation)
        else:
            self._schedule_drain(queue)

    def _subscribe(self, generation: int) -> None:
        try:
            unsubscribe = self._source.subscribe(
                lambda snapshot: self._publish(snapshot, generation)
            )
        except BaseException as error:
            _logger.exception("snapshot subscription failed")
            with self._lock:
                clients = tuple(self._subscribing_clients)
                self._subscribing = False
                self._subscribing_clients.clear()
                for client in clients:
                    self._terminate(client)
            if not isinstance(error, Exception):
                raise
            return
        callbacks: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[ExecutionSnapshot | None]]
        ] = []
        with self._lock:
            self._subscribing = False
            self._subscribing_clients.clear()
            if not self._replayed_during_subscription and self._latest is not None:
                for queue in self._clients:
                    self._pending[queue].append(self._latest)
            if self._clients:
                self._unsubscribe = unsubscribe
                callbacks = self._scheduled_clients()
            else:
                unsubscribe()
        self._run_callbacks(callbacks)

    def _remove(self, queue: asyncio.Queue[ExecutionSnapshot | None]) -> None:
        with self._lock:
            _ = self._clients.pop(queue, None)
            self._subscribing_clients.discard(queue)
            _ = self._pending.pop(queue, None)
            self._scheduled.discard(queue)
            if not self._clients and not self._subscribing:
                unsubscribe = self._unsubscribe
                self._unsubscribe = None
                if unsubscribe is not None:
                    self._generation += 1
                    unsubscribe()

    def _publish(self, snapshot: ExecutionSnapshot, generation: int) -> None:
        callbacks: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[ExecutionSnapshot | None]]
        ] = []
        with self._lock:
            if generation != self._generation:
                return
            self._latest = snapshot
            if self._subscribing:
                self._replayed_during_subscription = True
            for queue in tuple(self._clients):
                pending = self._pending[queue]
                if len(pending) >= _QUEUE_SIZE:
                    self._evict(queue)
                    continue
                pending.append(snapshot)
                if queue not in self._scheduled:
                    self._scheduled.add(queue)
                    callbacks.append((self._clients[queue], queue))
        self._run_callbacks(callbacks)

    def _schedule_drain(self, queue: asyncio.Queue[ExecutionSnapshot | None]) -> None:
        callbacks: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[ExecutionSnapshot | None]]
        ] = []
        with self._lock:
            if queue in self._clients and self._pending[queue] and queue not in self._scheduled:
                self._scheduled.add(queue)
                callbacks.append((self._clients[queue], queue))
        self._run_callbacks(callbacks)

    def _scheduled_clients(
        self,
    ) -> list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[ExecutionSnapshot | None]]]:
        callbacks: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[ExecutionSnapshot | None]]
        ] = []
        for queue, loop in self._clients.items():
            if self._pending[queue] and queue not in self._scheduled:
                self._scheduled.add(queue)
                callbacks.append((loop, queue))
        return callbacks

    def _run_callbacks(
        self,
        callbacks: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[ExecutionSnapshot | None]]],
    ) -> None:
        for loop, queue in callbacks:
            _ = loop.call_soon_threadsafe(self._deliver, queue)

    def _deliver(self, queue: asyncio.Queue[ExecutionSnapshot | None]) -> None:
        with self._lock:
            if queue not in self._clients:
                return
            self._scheduled.discard(queue)
            pending = self._pending[queue]
            while pending and not queue.full():
                queue.put_nowait(pending.popleft())

    def _evict(self, queue: asyncio.Queue[ExecutionSnapshot | None]) -> None:
        _logger.warning(
            "snapshot stream evicted on backpressure",
            extra={"event": "snapshot_backpressure_eviction", "capacity": _QUEUE_SIZE},
        )
        while True:
            try:
                _ = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        _ = self._pending.pop(queue, None)
        _ = queue.put_nowait(None)
        self._remove(queue)

    def _terminate(self, queue: asyncio.Queue[ExecutionSnapshot | None]) -> None:
        if queue not in self._clients:
            return
        self._evict(queue)


def _encode(snapshot: ExecutionSnapshot) -> str:
    payload = cast(object, msgspec.to_builtins(snapshot))
    return msgspec.json.encode(payload).decode()
