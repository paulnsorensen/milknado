"""Shared Textual presentation for immutable execution snapshots."""

from __future__ import annotations

from collections.abc import Callable
from threading import get_ident
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal
from textual.reactive import Reactive
from textual.widgets import DataTable, Footer, Header, Static
from typing_extensions import override

from milknado.app.run import ActiveRunSnapshot, ExecutionSnapshot, TerminalRunSnapshot
from milknado.app.run_panels import RunDetailPanel, RunListPanel
from milknado.app.run_source import ExecutionSnapshotSource
from milknado.app.run_view import events_text, subtitle_text
from milknado.domains.execution import RunLoopResult

if TYPE_CHECKING:
    from textual.events import Key, Resize

WIDE_MIN_COLUMNS = 116
RunSnapshot = ActiveRunSnapshot | TerminalRunSnapshot


class ExecutionSnapshotApp(App[RunLoopResult | None]):
    """Responsive presentation that depends only on immutable snapshots."""

    CSS: ClassVar[str] = """
    #workspace { height: 1fr; }
    #events { height: auto; max-height: 25%; margin: 0 1; border: round $secondary; }
    .visible { display: block; }
    .compact #workspace { display: block; }
    .compact #totals { display: block; }
    .compact #run-panel { width: 1fr; }
    .compact.list #detail { display: none; }
    .compact.detail #run-panel { display: none; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [  # noqa: V107 - Textual reads binding configuration
        ("up", "previous_run", "Previous run"),
        ("down", "next_run", "Next run"),
        ("enter", "open_detail", "Open selected run"),
        ("escape", "back", "Back"),
        ("r", "resume_output", "Resume output"),
        ("h", "help", "Help"),
        ("q", "quit_all", "Quit"),
    ]
    title: Reactive[str]
    sub_title: Reactive[str]

    def __init__(self, source: ExecutionSnapshotSource) -> None:
        super().__init__()
        self.source: ExecutionSnapshotSource = source
        self.snapshot: ExecutionSnapshot = source.snapshot()
        runs = self._runs()
        self.selected_run_id: str | None = runs[0].run_id if runs else None
        self.route: str = "list"
        self.compact: bool = False
        self.auto_follow: bool = True
        self._ui_thread_id: int | None = None
        self._unsubscribe: Callable[[], None] | None = None

    @override
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            yield RunListPanel(id="run-panel")
            yield RunDetailPanel(id="detail")
        yield Static(id="events", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self._ui_thread_id = get_ident()
        self._unsubscribe = self.source.subscribe(self._receive_snapshot)
        self._set_layout(self.size.width < WIDE_MIN_COLUMNS)
        self.show_snapshot(self.snapshot)

    def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def on_resize(self, event: Resize) -> None:
        self._set_layout(event.size.width < WIDE_MIN_COLUMNS)

    def show_snapshot(self, snapshot: ExecutionSnapshot) -> None:
        """Apply a replacement snapshot from the presentation source."""
        self._apply_snapshot(snapshot)

    def _receive_snapshot(self, snapshot: ExecutionSnapshot) -> None:
        """Marshal a source-thread replacement snapshot onto Textual's loop."""
        if get_ident() == self._ui_thread_id:
            self._apply_snapshot(snapshot)
        else:
            self.call_from_thread(self._apply_snapshot, snapshot)

    def _apply_snapshot(self, snapshot: ExecutionSnapshot) -> None:
        self.snapshot = snapshot
        runs = self._runs()
        if self.selected_run_id not in {run.run_id for run in runs}:
            self.selected_run_id = runs[0].run_id if runs else None
        self._refresh_view()

    def _sync_compact_route_to_focus(self) -> None:
        focused = self.screen.focused
        if focused is not None and focused.id == "guidance":
            self.route = "detail"

    def _set_layout(self, compact: bool) -> None:
        if compact:
            self._sync_compact_route_to_focus()
        self.compact = compact
        _ = self.set_class(compact, "compact")
        _ = self.set_class(compact and self.route == "list", "list")
        _ = self.set_class(compact and self.route == "detail", "detail")
        if not self.auto_follow:
            self.query_one("#detail", RunDetailPanel).preserve_output_offset()

    def _refresh_view(self) -> None:
        self.title = self.snapshot.goal
        self.sub_title = subtitle_text(self.snapshot)
        self.query_one("#run-panel", RunListPanel).update(
            self.snapshot, self._runs(), self.selected_run_id
        )
        self.query_one("#detail", RunDetailPanel).update(
            self._selected_run(),
            compact=self.compact,
            route=self.route,
            auto_follow=self.auto_follow,
        )
        self.query_one("#events", Static).update(events_text(self.snapshot.event_lines))

    def _runs(self) -> tuple[RunSnapshot, ...]:
        return (*self.snapshot.active_runs, *reversed(self.snapshot.terminal_runs))

    def _selected_run(self) -> RunSnapshot | None:
        return next((run for run in self._runs() if run.run_id == self.selected_run_id), None)

    def _selected_active_run(self) -> ActiveRunSnapshot | None:
        run = self._selected_run()
        return run if isinstance(run, ActiveRunSnapshot) else None

    def _run_index(self) -> int:
        runs = self._runs()
        ids = [run.run_id for run in runs]
        return ids.index(self.selected_run_id) if self.selected_run_id in ids else 0

    @on(DataTable.RowSelected, "#runs")
    def select_row(self, event: DataTable.RowSelected) -> None:
        self.selected_run_id = str(event.row_key.value)
        self._refresh_view()
        if self.compact:
            self.route = "detail"
            self._set_layout(True)

    def action_previous_run(self) -> None:
        self._move_selection(-1)

    def action_next_run(self) -> None:
        self._move_selection(1)

    def _move_selection(self, offset: int) -> None:
        runs = self._runs()
        if not runs:
            return
        self.selected_run_id = runs[(self._run_index() + offset) % len(runs)].run_id
        self._refresh_view()

    def action_open_detail(self) -> None:
        if self.compact and self._selected_run() is not None:
            self.route = "detail"
            self._set_layout(True)

    @override
    async def action_back(self) -> None:
        if self.compact and self.route == "detail":
            self.route = "list"
            self._set_layout(True)
        else:
            _ = self.query_one("#help", Static).remove_class("visible")

    def action_resume_output(self) -> None:
        self.auto_follow = True
        self._refresh_view()

    def action_help(self) -> None:
        _ = self.query_one("#help", Static).toggle_class("visible")

    def action_quit_all(self) -> None:
        self.exit()

    def pause_auto_follow(self) -> None:
        self.auto_follow = False
        self._refresh_view()

    def on_key(self, event: Key) -> None:
        if event.key in {"home", "end", "pageup", "pagedown"}:
            self.pause_auto_follow()
