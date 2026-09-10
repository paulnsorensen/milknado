"""Shared Textual presentation for immutable execution snapshots."""

from __future__ import annotations

from collections.abc import Callable
from threading import get_ident
from typing import TYPE_CHECKING, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import Reactive
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static
from typing_extensions import override

from milknado.app.run import ExecutionSnapshot
from milknado.app.run_panels import RunDetailPanel, RunListPanel
from milknado.app.run_source import ExecutionSnapshotSource
from milknado.app.run_view import (
    events_text,
    help_text,
    session_help_text,
    session_view,
    subtitle_text,
)
from milknado.app.session_commands import SessionCommandsMixin
from milknado.app.session_navigation import RunNavigationMixin
from milknado.domains.execution import RunLoopResult

if TYPE_CHECKING:
    from textual.events import Resize
    from textual.widget import Widget

WIDE_MIN_COLUMNS = 116


class _RunFooter(Footer):
    """Keep the compact open action visible despite DataTable's Enter binding."""

    @override
    def compose(self) -> ComposeResult:
        yield from super().compose()
        yield Static("Enter Open", id="open-hint", markup=False)


class _HelpScreen(ModalScreen[None]):
    SCOPED_CSS: ClassVar[bool] = False  # noqa: V107 - Textual class configuration
    AUTO_FOCUS: ClassVar[str | None] = "#help-scroll"  # noqa: V107 - Textual initial focus
    BINDINGS: ClassVar[list[BindingType]] = [  # noqa: V107 - Textual key bindings
        (key, "close_help", "Close") for key in ("escape", "f1", "?", "h", "q")
    ]
    DEFAULT_CSS: ClassVar[str] = """
    #session-help { align: center middle; background: $background 75%; }
    #help-scroll {
        width: 90%; max-width: 96; height: 90%; max-height: 32;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #help-close { dock: bottom; height: 1; text-align: center; background: $surface; }
    """

    def __init__(self, body: str) -> None:
        super().__init__(id="session-help")
        self._body: str = body

    @override
    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-scroll"):
            yield Static(self._body, id="help-overlay", markup=False)
        yield Static("↑/↓ scroll · Esc/F1 close", id="help-close", markup=False)

    def action_close_help(self) -> None:  # noqa: V105 - Textual binding action
        _ = self.dismiss(None)


class ExecutionSnapshotApp(SessionCommandsMixin, RunNavigationMixin, App[RunLoopResult | None]):
    """Responsive presentation for immutable execution and session snapshots."""

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
    .compact.list #open-hint { display: block; }
    #detail #help { display: none; }
    .compact #workspace { display: block; }
    .compact #totals { display: block; }
    .compact #run-panel { width: 1fr; }
    .compact.list #detail { display: none; }
    .compact.detail #run-panel { display: none; }
    .compact.detail #events, .compact #session-state { display: none; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [  # noqa: V107 - Textual reads binding configuration
        ("?", "help", "Help"),
        ("q", "quit_all", "Quit"),
        Binding("ctrl+c,ctrl+q", "quit_all", show=False, priority=True),
        ("e", "focus_events", "Events"),
        ("enter", "open_detail", "Open"),
        ("i", "focus_session", "Session input"),
        ("x", "focus_changes", "Changes"),
        Binding("up,k", "previous_run", show=False),
        Binding("down,j", "next_run", show=False),
        ("escape", "back", "Back"),
        ("r", "resume_output", "Resume output"),
        ("f1", "help", "Help"),
        ("h", "help", "Help"),
    ]
    title: Reactive[str]
    sub_title: Reactive[str]

    def __init__(self, source: ExecutionSnapshotSource, *, read_only: bool = False) -> None:
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
        self._init_session_ui(read_only=read_only)

    @override
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            yield RunListPanel(id="run-panel")
            yield RunDetailPanel(id="detail")
        with VerticalScroll(id="events") as events:
            events.border_title = "Events"
            yield Static(id="events-text", markup=False)
        yield _RunFooter()

    def on_mount(self) -> None:
        self._ui_thread_id = get_ident()
        self._unsubscribe = self.source.subscribe(self._receive_snapshot)
        self.set_layout(self.size.width < WIDE_MIN_COLUMNS)
        self.refresh_view()
        _ = self.set_interval(1.0, self.refresh_session_changes)

    def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def on_resize(self, event: Resize) -> None:
        compact = event.size.width < WIDE_MIN_COLUMNS
        changed = compact != self.compact
        self.set_layout(compact)
        if changed and self.screen.is_mounted:
            self.refresh_view()

    @override
    async def action_quit(self) -> None:
        self.action_quit_all()

    def action_help(self) -> None:  # noqa: V105 - Textual binding action
        selected = self.selected_run()
        focus = self.screen.focused
        body = help_text(
            selected,
            compact=self.compact,
            route=self.route,
            auto_follow=self.auto_follow,
        )
        session = session_view(selected)
        if session.context is not None:
            body += "\nx changed files"
        if not self.read_only and session.actions:
            body += f"\ni session input\n{session_help_text(session)}"
        _ = self.push_screen(
            _HelpScreen(body),
            lambda _: self._restore_focus_after_help(focus),
        )

    def _restore_focus_after_help(self, focus: Widget | None) -> None:
        _ = self.call_after_refresh(self._restore_focus, focus)

    def _restore_focus(self, focus: Widget | None) -> None:
        if not self.screen.is_modal:
            self.set_focus(focus if focus and focus.region.area and not focus.disabled else None)

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
        if not self.screen.is_mounted:
            return
        self.refresh_view()

    def _sync_compact_route_to_focus(self) -> None:
        focused = self.screen.focused
        if focused is not None and focused.id in {
            "guidance",
            "session-input",
            "session-action",
            "session-permission",
            "session-submit",
            "changes-files",
        }:
            self.route = "detail"

    def set_layout(self, compact: bool) -> None:
        if compact:
            self._sync_compact_route_to_focus()
        self.compact = compact
        _ = self.set_class(compact, "compact")
        _ = self.set_class(compact and self.route == "list", "list")
        _ = self.set_class(compact and self.route == "detail", "detail")
        if not self.auto_follow:
            self.query_one("#detail", RunDetailPanel).preserve_output_offset()

    def refresh_view(self) -> None:
        self.title = self.snapshot.goal
        self.sub_title = subtitle_text(self.snapshot)
        selected = self.selected_run()
        self.query_one("#run-panel", RunListPanel).update(
            self.snapshot, self._runs(), self.selected_run_id
        )
        self.query_one("#detail", RunDetailPanel).update(
            selected,
            auto_follow=self.auto_follow,
        )
        self.query_one("#events-text", Static).update(
            events_text(self.snapshot.event_lines, self.snapshot.listener_errors)
        )
        self.refresh_session_changes()
