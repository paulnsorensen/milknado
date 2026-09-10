"""Textual observer for durable execution snapshots."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from typing_extensions import override

from milknado.app.run import ExecutionSnapshot
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.app.watch import WatchSnapshotSource

POLL_INTERVAL_SECONDS = 1.0


class SnapshotSource(Protocol):
    def snapshot(self) -> ExecutionSnapshot: ...


class _WatchController:
    """Snapshot-only adapter for the shared execution view."""

    def __init__(self, source: SnapshotSource) -> None:
        self.source: SnapshotSource = source

    def snapshot(self) -> ExecutionSnapshot:
        return self.source.snapshot()

    @staticmethod
    def subscribe(
        listener: Callable[[ExecutionSnapshot], None],
    ) -> Callable[[], None]:
        del listener
        return lambda: None


class WatchApp(ExecutionSnapshotApp):
    """Read-only execution view refreshed from durable state."""

    def __init__(
        self,
        source: SnapshotSource,
        *,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self.poll_interval: float = poll_interval
        super().__init__(_WatchController(source), read_only=True)

    @override
    def on_mount(self) -> None:  # noqa: V105 - Textual lifecycle handler
        _ = self.set_interval(self.poll_interval, self.poll)

    def poll(self) -> None:
        self.show_snapshot(self.source.snapshot())


def run_watch_tui(project_root: Path, db_path: Path) -> None:
    """Run the read-only Textual observer until the user quits."""
    _ = WatchApp(WatchSnapshotSource(project_root, db_path)).run()
