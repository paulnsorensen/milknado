from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from typing import Protocol


class _GraphHandle(Protocol):
    @property
    def synchronization_lock(self) -> AbstractContextManager[object]: ...

    @property
    def _conn(self) -> sqlite3.Connection: ...


class SubFacade:
    def __init__(self, graph: _GraphHandle) -> None:
        self._graph: _GraphHandle = graph

    @property
    def synchronization_lock(self) -> AbstractContextManager[object]:
        return self._graph.synchronization_lock

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._graph._conn  # pyright: ignore[reportPrivateUsage]
