"""Thread-safe fan-out for execution snapshots."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from typing import cast

import msgspec

from milknado.app.run_source import ExecutionSnapshot, ExecutionSnapshotSource


class SnapshotFanout:
    """Bridge source callbacks into one queue per connected stream."""

    def __init__(self, source: ExecutionSnapshotSource) -> None:
        self._source: ExecutionSnapshotSource = source
        self._clients: dict[asyncio.Queue[ExecutionSnapshot], asyncio.AbstractEventLoop] = {}
        self._unsubscribe: Callable[[], None] | None = None
        self._lock: threading.Lock = threading.Lock()

    async def events(self) -> AsyncIterator[dict[str, str]]:
        queue: asyncio.Queue[ExecutionSnapshot] = asyncio.Queue()
        self._add(queue, asyncio.get_running_loop())
        try:
            while True:
                snapshot = await queue.get()
                yield {"event": "snapshot", "data": _encode(snapshot)}
        finally:
            self._remove(queue)

    def _add(
        self,
        queue: asyncio.Queue[ExecutionSnapshot],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        with self._lock:
            if not self._clients:
                self._unsubscribe = self._source.subscribe(self._publish)
            self._clients[queue] = loop

    def _remove(self, queue: asyncio.Queue[ExecutionSnapshot]) -> None:
        unsubscribe: Callable[[], None] | None = None
        with self._lock:
            _ = self._clients.pop(queue, None)
            if not self._clients:
                unsubscribe = self._unsubscribe
                self._unsubscribe = None
        if unsubscribe is not None:
            unsubscribe()

    def _publish(self, snapshot: ExecutionSnapshot) -> None:
        with self._lock:
            clients = tuple(self._clients.items())
        for queue, loop in clients:
            _ = loop.call_soon_threadsafe(queue.put_nowait, snapshot)


def _encode(snapshot: ExecutionSnapshot) -> str:
    payload = cast(object, msgspec.to_builtins(snapshot))
    return msgspec.json.encode(payload).decode()
