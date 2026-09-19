"""Thread-safe fan-out for execution snapshots."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable
from typing import cast

import msgspec

from milknado.app.run_source import ExecutionSnapshot, ExecutionSnapshotSource

_QUEUE_SIZE = 32


class SnapshotFanout:
    """Bridge source callbacks into one bounded queue per connected stream."""

    def __init__(self, source: ExecutionSnapshotSource) -> None:
        self._source: ExecutionSnapshotSource = source
        self._clients: dict[
            asyncio.Queue[ExecutionSnapshot | None], asyncio.AbstractEventLoop
        ] = {}
        self._pending: dict[asyncio.Queue[ExecutionSnapshot | None], ExecutionSnapshot] = {}
        self._scheduled: set[asyncio.Queue[ExecutionSnapshot | None]] = set()
        self._unsubscribe: Callable[[], None] | None = None
        self._subscribing: bool = False
        self._lock: threading.Lock = threading.Lock()

    async def events(self) -> AsyncGenerator[dict[str, str], None]:
        queue: asyncio.Queue[ExecutionSnapshot | None] = asyncio.Queue(_QUEUE_SIZE)
        self._add(queue, asyncio.get_running_loop())
        try:
            while (snapshot := await queue.get()) is not None:
                yield {"event": "snapshot", "data": _encode(snapshot)}
        finally:
            self._remove(queue)

    def _add(
        self,
        queue: asyncio.Queue[ExecutionSnapshot | None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        subscribe = False
        with self._lock:
            self._clients[queue] = loop
            if self._unsubscribe is None and not self._subscribing:
                self._subscribing = True
                subscribe = True
        if not subscribe:
            return
        try:
            unsubscribe = self._source.subscribe(self._publish)
        except BaseException:
            with self._lock:
                self._subscribing = False
                _ = self._clients.pop(queue, None)
            raise
        remove_subscription = False
        with self._lock:
            self._subscribing = False
            if self._clients:
                self._unsubscribe = unsubscribe
            else:
                remove_subscription = True
        if remove_subscription:
            unsubscribe()

    def _remove(self, queue: asyncio.Queue[ExecutionSnapshot | None]) -> None:
        unsubscribe: Callable[[], None] | None = None
        with self._lock:
            _ = self._clients.pop(queue, None)
            _ = self._pending.pop(queue, None)
            self._scheduled.discard(queue)
            if not self._clients and not self._subscribing:
                unsubscribe = self._unsubscribe
                self._unsubscribe = None
        if unsubscribe is not None:
            unsubscribe()

    def _publish(self, snapshot: ExecutionSnapshot) -> None:
        callbacks: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[ExecutionSnapshot | None]]
        ] = []
        with self._lock:
            for queue, loop in self._clients.items():
                self._pending[queue] = snapshot
                if queue not in self._scheduled:
                    self._scheduled.add(queue)
                    callbacks.append((loop, queue))
        for loop, queue in callbacks:
            _ = loop.call_soon_threadsafe(self._deliver, queue)

    def _deliver(self, queue: asyncio.Queue[ExecutionSnapshot | None]) -> None:
        with self._lock:
            if queue not in self._clients:
                return
            snapshot = self._pending.pop(queue, None)
            self._scheduled.discard(queue)
        if snapshot is None:
            return
        self._enqueue(queue, snapshot)

    def _enqueue(
        self,
        queue: asyncio.Queue[ExecutionSnapshot | None],
        snapshot: ExecutionSnapshot,
    ) -> None:
        with self._lock:
            if queue not in self._clients:
                return
        try:
            queue.put_nowait(snapshot)
        except asyncio.QueueFull:
            _ = queue.get_nowait()
            queue.put_nowait(None)
            self._remove(queue)


def _encode(snapshot: ExecutionSnapshot) -> str:
    payload = cast(object, msgspec.to_builtins(snapshot))
    return msgspec.json.encode(payload).decode()
