"""Minimal snapshot contract for shared execution presentation."""

from collections.abc import Callable
from typing import Protocol

from milknado.app.run import ExecutionSnapshot


class ExecutionSnapshotSource(Protocol):
    def snapshot(self) -> ExecutionSnapshot: ...

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]: ...
