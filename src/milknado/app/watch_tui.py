"""Textual observer for durable execution snapshots."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Protocol

from textual.binding import Binding, BindingType
from typing_extensions import override

from milknado.app.run import ExecutionSnapshot
from milknado.app.run_source import NodeSnapshotRequest
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.app.watch import AttachedWatchSource, WatchSnapshotSource
from milknado.domains.common import SessionInput
from milknado.domains.graph import NodeDetailResponse

POLL_INTERVAL_SECONDS = 1.0


class SnapshotSource(Protocol):
    def snapshot(self) -> ExecutionSnapshot: ...

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse: ...


class _WatchController:
    """Snapshot-only adapter for the shared execution view."""

    def __init__(self, source: SnapshotSource) -> None:
        self.source: SnapshotSource = source

    def snapshot(self) -> ExecutionSnapshot:
        return self.source.snapshot()

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        return self.source.node_snapshot(request)

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if callable(close):
            _ = close()

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        if isinstance(self.source, AttachedWatchSource):
            return self.source.session_input(run_id, command)
        return False

    @staticmethod
    def subscribe(
        listener: Callable[[ExecutionSnapshot], None],
    ) -> Callable[[], None]:
        del listener
        return lambda: None


class WatchApp(ExecutionSnapshotApp):
    """Read-only execution view refreshed from durable state."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("?", "help", "Help"),
        ("q", "quit_all", "Quit"),
        Binding("ctrl+c,ctrl+q", "quit_all", show=False, priority=True),
        ("e", "focus_events", "Events"),
        ("enter", "open_detail", "Open"),
        ("x", "focus_changes", "Changes"),
        Binding("up,k", "previous_run", show=False),
        Binding("down,j", "next_run", show=False),
        ("escape", "back", "Back"),
        ("r", "resume_output", "Resume output"),
        ("f1", "help", "Help"),
        ("h", "help", "Help"),
    ]

    def __init__(
        self,
        source: SnapshotSource,
        *,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        read_only: bool = True,
    ) -> None:
        self.poll_interval: float = poll_interval
        controller = _WatchController(source)
        self.controller: _WatchController = controller
        super().__init__(controller, read_only=read_only)
        if not read_only:
            self.bind("i", "focus_session", description="Session input")

    @override
    def on_mount(self) -> None:  # noqa: V105 - Textual lifecycle handler
        super().on_mount()
        _ = self.set_interval(self.poll_interval, self.poll)

    @override
    def on_unmount(self) -> None:  # noqa: V105 - Textual lifecycle handler
        super().on_unmount()
        close = getattr(self.source, "close", None)
        if callable(close):
            _ = close()

    def poll(self) -> None:
        self.show_snapshot(self.source.snapshot())


def run_watch_tui(project_root: Path, db_path: Path) -> None:
    """Run the read-only Textual observer until the user quits."""
    _ = WatchApp(WatchSnapshotSource(project_root, db_path)).run()


def run_attached_watch_tui(source: AttachedWatchSource) -> None:
    """Run the attached watch with explicit command admission enabled."""
    _ = WatchApp(source, read_only=False).run()
