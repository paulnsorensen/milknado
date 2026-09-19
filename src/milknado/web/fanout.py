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
        self._latest: ExecutionSnapshot | None = None
        self._subscribing: bool = False
        self._subscribing_clients: set[asyncio.Queue[ExecutionSnapshot | None]] = set()
        self._replayed_during_subscription: bool = False
        self._lock: threading.RLock = threading.RLock()

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
            if self._subscribing:
                self._subscribing_clients.add(queue)
                if self._replayed_during_subscription and self._latest is not None:
                    queue.put_nowait(self._latest)
            elif self._unsubscribe is not None and self._latest is not None:
                queue.put_nowait(self._latest)
            if self._unsubscribe is None and not self._subscribing:
                self._subscribing = True
                self._subscribing_clients = {queue}
                self._replayed_during_subscription = False
                subscribe = True
        if not subscribe:
            return
        try:
            unsubscribe = self._source.subscribe(self._publish)
        except BaseException as error:
            with self._lock:
                self._subscribing = False
                failed_clients = tuple(self._subscribing_clients)
                self._subscribing_clients.clear()
                for client in failed_clients:
                    self._terminate(client)
            if not isinstance(error, Exception):
                raise
            return
        with self._lock:
            self._subscribing = False
            self._subscribing_clients.clear()
            if not self._replayed_during_subscription and self._latest is not None:
                for client in self._clients:
                    client.put_nowait(self._latest)
            if self._clients:
                self._unsubscribe = unsubscribe
            else:
                unsubscribe()

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
                    unsubscribe()

    def _publish(self, snapshot: ExecutionSnapshot) -> None:
        callbacks: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[ExecutionSnapshot | None]]
        ] = []
        with self._lock:
            self._latest = snapshot
            if self._subscribing:
                self._replayed_during_subscription = True
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
            while True:
                try:
                    _ = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            queue.put_nowait(None)
            self._remove(queue)

    def _terminate(self, queue: asyncio.Queue[ExecutionSnapshot | None]) -> None:
        if queue not in self._clients:
            return
        while True:
            try:
                _ = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queue.put_nowait(None)
        self._remove(queue)


def _encode(snapshot: ExecutionSnapshot) -> str:
    payload = cast(object, msgspec.to_builtins(snapshot))
    return msgspec.json.encode(payload).decode()
