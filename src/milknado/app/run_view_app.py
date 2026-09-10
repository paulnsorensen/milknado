"""Shared Textual presentation for immutable execution snapshots."""

from __future__ import annotations

from collections.abc import Callable
from threading import get_ident
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import Reactive
from textual.widgets import DataTable, Footer, Header, Static
from typing_extensions import override

from milknado.app.run import ActiveRunSnapshot, ExecutionSnapshot, TerminalRunSnapshot
from milknado.app.run_panels import RunDetailPanel, RunListPanel
from milknado.app.run_source import ExecutionSnapshotSource
from milknado.app.run_view import events_text, help_text, subtitle_text
from milknado.domains.execution import RunLoopResult

if TYPE_CHECKING:
    from textual.events import Key, Resize

WIDE_MIN_COLUMNS = 116
RunSnapshot = ActiveRunSnapshot | TerminalRunSnapshot


class _RunFooter(Footer):
    """Keep the compact open action visible despite DataTable's Enter binding."""

    @override
    def compose(self) -> ComposeResult:
        yield from super().compose()
        yield Static("Enter Open", id="open-hint", markup=False)


class ExecutionSnapshotApp(App[RunLoopResult | None]):
    """Responsive presentation that depends only on immutable snapshots."""

    AUTO_FOCUS: ClassVar[str | None] = "#runs"  # noqa: V107 - Textual reads initial focus

    CSS: ClassVar[str] = """
    Screen { layers: base overlay; }
    #workspace { height: 1fr; layer: base; }
    #events {
        height: auto; min-height: 4; max-height: 5;
        margin: 0 1; border: round $secondary; layer: base;
    }
    Header, Footer { layer: base; }
    #open-hint {
        display: none;
        dock: right;
        width: auto;
        height: 1;
        padding: 0 1;
        background: $footer-background;
    }
    .compact #open-hint { display: block; }
    #help-overlay {
        display: none;
        position: absolute;
        offset: 0 1;
        margin: 0 1 2 1;
        width: 1fr;
        max-height: 1fr;
        padding: 1 2;
        border: round $accent;
        background: $surface;
        layer: overlay;
        overflow-y: auto;
    }
    #help-overlay.visible { display: block; }
    #detail #help { display: none; }
    .compact #workspace { display: block; }
    .compact #totals { display: block; }
    .compact #run-panel { width: 1fr; }
    .compact.list #detail { display: none; }
    .compact.detail #run-panel { display: none; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [  # noqa: V107 - Textual reads binding configuration
        ("?", "help", "Help"),
        ("q", "quit_all", "Quit"),
        Binding("ctrl+c,ctrl+q", "quit_all", show=False, priority=True),
        ("e", "focus_events", "Events"),
        ("enter", "open_detail", "Open"),
        ("up", "previous_run", "Previous run"),
        ("down", "next_run", "Next run"),
        ("j", "next_run", "Next run"),
        ("k", "previous_run", "Previous run"),
        ("escape", "back", "Back"),
        ("r", "resume_output", "Resume output"),
        ("f1", "help", "Help"),
        ("h", "help", "Help"),
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
        with VerticalScroll(id="events") as events:
            events.border_title = "Events"
            yield Static(id="events-text", markup=False)
        yield Static(id="help-overlay", markup=False)
        yield _RunFooter()

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
            auto_follow=self.auto_follow,
        )
        self.query_one("#help-overlay", Static).update(
            help_text(
                self._selected_run(),
                compact=self.compact,
                route=self.route,
                auto_follow=self.auto_follow,
            )
        )
        self.query_one("#events-text", Static).update(
            events_text(self.snapshot.event_lines, self.snapshot.listener_errors)
        )

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
        if self.compact:
            self.action_open_detail()
        else:
            self._refresh_view()

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
            self.set_focus(None)
            self.route = "detail"
            self._set_layout(True)
            self._refresh_view()

    @override
    async def action_back(self) -> None:
        help_overlay = self.query_one("#help-overlay", Static)
        if help_overlay.has_class("visible"):
            _ = help_overlay.remove_class("visible")
            return
        self.set_focus(None)
        if self.compact and self.route == "detail":
            self.route = "list"
            self._set_layout(True)
            self._refresh_view()
        _ = self.call_after_refresh(self.query_one("#runs", DataTable).focus)

    def action_focus_events(self) -> None:
        _ = self.query_one("#events", VerticalScroll).focus()

    def action_resume_output(self) -> None:
        self.auto_follow = True
        self._refresh_view()

    def action_help(self) -> None:
        _ = self.query_one("#help-overlay", Static).toggle_class("visible")

    def action_quit_all(self) -> None:
        self.exit()

    def pause_auto_follow(self) -> None:
        self.auto_follow = False
        self._refresh_view()

    def on_key(self, event: Key) -> None:
        if (
            event.key in {"home", "end", "pageup", "pagedown"}
            and not self.query_one("#events", VerticalScroll).has_focus
        ):
            self.pause_auto_follow()
