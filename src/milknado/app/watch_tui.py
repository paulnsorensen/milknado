"""Textual observer for durable execution snapshots."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import get_ident
from typing import Protocol, cast

from milknado.app.run import ExecutionController, ExecutionSnapshot
from milknado.app.run_tui import WIDE_MIN_COLUMNS, ExecutionApp
from milknado.app.watch import WatchSnapshotSource

POLL_INTERVAL_SECONDS = 1.0


class SnapshotSource(Protocol):
    def snapshot(self) -> ExecutionSnapshot: ...


class _WatchController:
    """Snapshot-only adapter for the shared execution view."""

    def __init__(self, source: SnapshotSource) -> None:
        self.source = source

    def snapshot(self) -> ExecutionSnapshot:
        return self.source.snapshot()

    @staticmethod
    def subscribe(
        _listener: Callable[[ExecutionSnapshot], None],
    ) -> Callable[[], None]:
        return lambda: None


class WatchApp(ExecutionApp):
    """Read-only execution view refreshed from durable state."""

    BINDINGS = [  # noqa: V107 - Textual reads binding configuration
        ("up", "previous_run", "Previous run"),
        ("down", "next_run", "Next run"),
        ("enter", "open_detail", "Open selected run"),
        ("escape", "back", "Back"),
        ("r", "resume_output", "Resume output"),
        ("h", "help", "Help"),
        ("q", "quit_all", "Quit"),
    ]

    def __init__(
        self,
        source: SnapshotSource,
        *,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self.source = source
        self.poll_interval = poll_interval
        controller = _WatchController(source)
        super().__init__(cast(ExecutionController, controller))

    def on_mount(self) -> None:  # noqa: V105 - Textual lifecycle handler
        self._ui_thread_id = get_ident()
        self._set_layout(self.size.width < WIDE_MIN_COLUMNS)
        self._refresh_view()
        self.set_interval(self.poll_interval, self.poll)

    def poll(self) -> None:
        self._apply_snapshot(self.source.snapshot())

    def action_quit_all(self) -> None:  # noqa: V105 - Textual binding action
        self.exit()


def run_watch_tui(project_root: Path, db_path: Path) -> None:
    """Run the read-only Textual observer until the user quits."""
    return WatchApp(WatchSnapshotSource(project_root, db_path)).run()
