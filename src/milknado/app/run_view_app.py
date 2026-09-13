"""Shared Textual presentation for immutable execution snapshots."""

from __future__ import annotations

from collections.abc import Callable
from threading import get_ident
from typing import TYPE_CHECKING, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import Reactive
from textual.widgets import Header, Static
from typing_extensions import override

from milknado.app.graph_navigation import GraphSelectionMixin
from milknado.app.run import ActiveRunSnapshot, ExecutionSnapshot, TerminalRunSnapshot
from milknado.app.run_layout import MINIMUM_FALLBACK_TEXT, RUN_VIEW_CSS, WIDE_MIN_COLUMNS
from milknado.app.run_overlays import HelpScreen, RunFooter
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
from milknado.domains.graph import NodeDetailResponse

if TYPE_CHECKING:
    from textual.events import Resize
    from textual.widget import Widget


class ExecutionSnapshotApp(
    GraphSelectionMixin, SessionCommandsMixin, RunNavigationMixin, App[RunLoopResult | None]
):
    """Responsive presentation for immutable execution and session snapshots."""

    AUTO_FOCUS: ClassVar[str | None] = "#runs"  # noqa: V107 - Textual reads initial focus

    CSS: ClassVar[str] = RUN_VIEW_CSS

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
        Binding("[", "previous_detail_page", show=False),
        Binding("]", "next_detail_page", show=False),
        Binding("(", "previous_history_page", show=False),
        Binding(")", "next_history_page", show=False),
        ("escape", "back", "Back"),
        Binding("r", "resume_output", show=False),
        ("f1", "help", "Help"),
        Binding("h", "help", show=False),
    ]
    title: Reactive[str]
    sub_title: Reactive[str]

    def __init__(self, source: ExecutionSnapshotSource, *, read_only: bool = False) -> None:
        super().__init__()
        self.source: ExecutionSnapshotSource = source
        self.snapshot: ExecutionSnapshot = source.snapshot()
        runs = self._runs()
        self.selected_run_id: str | None = runs[0].run_id if runs else None
        graph = self.snapshot.graph
        self.selected_node_id: int | None = (
            runs[0].node_id
            if runs
            else graph.root_ids[0]
            if graph is not None and graph.root_ids
            else graph.nodes[0].id
            if graph is not None and graph.nodes
            else None
        )
        self.route: str = "list"
        self.compact: bool = False
        self.minimum: bool = False
        self.auto_follow: bool = True
        self._minimum_focus: Widget | None = None
        self._ui_thread_id: int | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self.node_detail: NodeDetailResponse | None = self.snapshot.node
        self.node_request_generation: int = 0
        self.detail_page: int = 0
        self.session_event_page: int = 0
        self._init_session_ui(read_only=read_only)

    def runs(self) -> tuple[ActiveRunSnapshot | TerminalRunSnapshot, ...]:
        return self._runs()

    @override
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            yield RunListPanel(id="run-panel")
            yield RunDetailPanel(id="detail")
        with VerticalScroll(id="events") as events:
            events.border_title = "Events"
            yield Static(id="events-text", markup=False)
        yield Static(MINIMUM_FALLBACK_TEXT, id="minimum-fallback", markup=False)
        yield RunFooter()

    def on_mount(self) -> None:
        self._ui_thread_id = get_ident()
        self._unsubscribe = self.source.subscribe(self._receive_snapshot)
        self.set_layout(
            self.size.width < WIDE_MIN_COLUMNS,
            minimum=self.size.width < 60 or self.size.height < 18,
        )
        self.refresh_view()
        if (
            self.selected_node_id is not None
            and self.node_detail is None
            and self.node_request_generation == 0
        ):
            self._request_node_detail()
        _ = self.set_interval(1.0, self.refresh_session_changes)
        if self.snapshot.graph is not None and not self.minimum:
            _ = self.call_after_refresh(self.focus_initial_selector)

    def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def on_resize(self, event: Resize) -> None:
        compact = event.size.width < WIDE_MIN_COLUMNS
        minimum = event.size.width < 60 or event.size.height < 18
        changed = compact != self.compact or minimum != self.minimum
        self.set_layout(compact, minimum=minimum)
        if changed and self.screen.is_mounted:
            self.refresh_view()

    @override
    async def action_quit(self) -> None:
        self.action_quit_all()

    def action_help(self) -> None:  # noqa: V105 - Textual binding action
        selected = self.selected_run()
        focus = self.screen.focused
        if self.minimum:
            body = f"Help\n{MINIMUM_FALLBACK_TEXT}"
        else:
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
            HelpScreen(body),
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
        old_node_id, old_graph = self.selected_node_id, self.snapshot.graph
        focused_id = self.screen.focused.id if self.screen.focused is not None else None
        selected_run_id = self.selected_run_id
        self.snapshot = snapshot
        runs = self._runs()
        self.selected_node_id, self.selected_run_id = self.reconcile_graph_selection(
            runs, selected_run_id
        )
        if snapshot.node is not None and snapshot.node.matches(
            self.selected_node_id or -1, self.node_request_generation
        ):
            self.node_detail = snapshot.node
        elif snapshot.graph is not None and self.selected_node_id is None:
            self.node_detail = None
        if self.selected_node_id != old_node_id:
            self.detail_page = 0
            self.session_event_page = 0
            self._request_node_detail()
        elif self.selected_node_id is not None:
            self._request_node_detail(clear_detail=False)
        if not self.screen.is_mounted:
            return
        self.refresh_view()
        if (
            not self.minimum
            and self._minimum_focus is None
            and snapshot.graph is not None
            and (old_graph is None or focused_id in {None, "runs"})
        ):
            _ = self.call_after_refresh(self.focus_initial_selector)

    def _sync_compact_route_to_focus(self, focused: Widget | None = None) -> None:
        focused = focused or self.screen.focused
        if focused is not None and focused.id in {
            "guidance",
            "session-input",
            "session-action",
            "session-permission",
            "session-submit",
            "changes-files",
        }:
            self.route = "detail"

    def set_layout(self, compact: bool, *, minimum: bool = False) -> None:
        entering_minimum = minimum and not self.minimum
        leaving_minimum = self.minimum and not minimum
        if entering_minimum and not self.screen.is_modal:
            focus = self.screen.focused
            self._minimum_focus = (
                focus if focus is not None and focus.is_mounted and not focus.disabled else None
            )
            _ = self.set_focus(None)
        if compact and not minimum:
            self._sync_compact_route_to_focus(
                self._minimum_focus if leaving_minimum else self.screen.focused
            )
        self.compact = compact
        self.minimum = minimum
        _ = self.set_class(compact, "compact")
        _ = self.set_class(compact and self.route == "list", "list")
        _ = self.set_class(compact and self.route == "detail", "detail")
        _ = self.set_class(minimum, "minimum")
        if leaving_minimum:
            _ = self.call_after_refresh(self._restore_minimum_focus, self._minimum_focus)
        if not self.auto_follow:
            self.query_one("#detail", RunDetailPanel).preserve_output_offset()

    def _restore_minimum_focus(self, focus: Widget | None, *, retry: bool = True) -> None:
        if self.minimum or self.screen.is_modal:
            return
        if focus is not None and focus.is_mounted and not focus.disabled:
            if focus.region.area:
                self._minimum_focus = None
                self.set_focus(focus)
            elif retry:
                _ = self.call_after_refresh(self._restore_minimum_focus, focus, retry=False)
            else:
                self._minimum_focus = None
                self.focus_initial_selector()
        else:
            self._minimum_focus = None
            self.focus_initial_selector()

    def refresh_view(self) -> None:
        self.title = self.snapshot.goal
        self.sub_title = subtitle_text(self.snapshot)
        selected = self.selected_run()
        self.query_one("#run-panel", RunListPanel).update(
            self.snapshot, self._runs(), self.selected_run_id, self.selected_node_id
        )
        self.query_one("#detail", RunDetailPanel).update(
            selected,
            node=self.selected_node(),
            node_detail=self.node_detail.detail if self.node_detail is not None else None,
            auto_follow=self.auto_follow,
        )
        self.query_one("#events-text", Static).update(
            events_text(self.snapshot.event_lines, self.snapshot.listener_errors)
        )
        self.refresh_session_changes()
