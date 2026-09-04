"""Textual panels for the execution TUI — render snapshot data via run_view."""

from __future__ import annotations

from typing import Protocol, cast, final

from rich.console import RenderableType
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp
from textual.widgets import DataTable, Input, Static
from typing_extensions import override

from milknado.app.run import ActiveRunSnapshot, ExecutionSnapshot, TerminalRunSnapshot
from milknado.app.run_view import (
    actions_text,
    help_text,
    output_body,
    output_border_title,
    run_index,
    run_row,
    subtitle_text,
    summary_text,
)


class _ExecutionAppLike(Protocol):
    compact: bool
    route: str

    def pause_auto_follow(self) -> None: ...


RunSnapshot = ActiveRunSnapshot | TerminalRunSnapshot
RUN_COLUMNS = (("Node", 3), ("Description", 20), ("Status", 12), ("Progress", 13), ("Elapsed", 6))


class RunTable(DataTable[RenderableType]):
    """Run selector that preserves the focused table across compact resizes."""

    def on_focus(self) -> None:
        app = cast(_ExecutionAppLike, cast(object, self.app))
        if not app.compact:
            app.route = "list"


@final
class RunListPanel(Vertical):
    """List pane owning the totals line and the run table."""

    DEFAULT_CSS = """
    RunListPanel { width: 68; height: 1fr; }
    #totals { height: auto; margin: 0 1; display: none; }
    #runs { height: 1fr; }
    """

    @override
    def compose(self) -> ComposeResult:
        yield Static(id="totals", markup=False)
        yield RunTable(id="runs", cursor_type="row")

    def on_mount(self) -> None:
        table = cast(DataTable[RenderableType], self.query_one("#runs", DataTable))
        for label, width in RUN_COLUMNS:
            _ = table.add_column(label, width=width)

    def update(
        self,
        snapshot: ExecutionSnapshot,
        runs: tuple[RunSnapshot, ...],
        selected_run_id: str | None,
    ) -> None:
        self.query_one("#totals", Static).update(subtitle_text(snapshot))
        table = cast(DataTable[RenderableType], self.query_one("#runs", DataTable))
        _ = table.clear(columns=False)
        for run in runs:
            _ = table.add_row(*run_row(run), key=run.run_id)
        if selected_run_id is not None:
            _ = table.move_cursor(row=run_index(runs, selected_run_id))


@final
class RunDetailPanel(VerticalScroll):
    """Detail pane that pauses output following before consuming wheel input."""

    DEFAULT_CSS = """
    RunDetailPanel { width: 1fr; height: 1fr; }
    #output, #actions, #help, #confirmation { margin: 0 1; }
    #output { height: 1fr; overflow-y: auto; border: round $primary; }
    #confirmation { display: none; color: $warning; }
    #help { display: none; }
    #guidance { margin: 0 1 1 1; }
    """

    @override
    def compose(self) -> ComposeResult:
        yield Static(id="summary", markup=False)
        with VerticalScroll(id="output"):
            yield Static(id="output-text", markup=False)
        yield Static(id="actions", markup=False)
        yield Static(id="help", markup=False)
        yield Static(id="confirmation", markup=False)
        yield Input(placeholder="Queue guidance for the selected run", id="guidance")

    def on_mouse_scroll_up(self, _event: MouseScrollUp) -> None:
        cast(_ExecutionAppLike, cast(object, self.app)).pause_auto_follow()

    def on_mouse_scroll_down(self, _event: MouseScrollDown) -> None:
        cast(_ExecutionAppLike, cast(object, self.app)).pause_auto_follow()

    def update(
        self,
        selected: RunSnapshot | None,
        *,
        compact: bool,
        route: str,
        auto_follow: bool,
    ) -> None:
        self.query_one("#summary", Static).update(summary_text(selected))
        self.query_one("#output", VerticalScroll).border_title = output_border_title(
            auto_follow=auto_follow
        )
        self.query_one("#output-text", Static).update(output_body(selected))
        self.query_one("#actions", Static).update(actions_text(selected))
        self.query_one("#help", Static).update(
            help_text(selected, compact=compact, route=route, auto_follow=auto_follow)
        )
        active = selected if isinstance(selected, ActiveRunSnapshot) else None
        self.query_one("#guidance", Input).disabled = (
            active is None or not active.actions.can_queue_guidance
        )
        if auto_follow:
            self.query_one("#output", VerticalScroll).scroll_end(animate=False)

    def preserve_output_offset(self) -> None:
        """Re-apply the paused output position: a relayout otherwise scrolls it to the top."""
        output = self.query_one("#output", VerticalScroll)
        position = output.scroll_offset.y
        _ = self.call_after_refresh(output.scroll_to, y=position, animate=False)

    def set_confirmation(self, message: str) -> None:
        confirmation = self.query_one("#confirmation", Static)
        confirmation.update(message)
        _ = confirmation.add_class("visible")

    def clear_confirmation(self) -> None:
        _ = self.query_one("#confirmation", Static).remove_class("visible")
