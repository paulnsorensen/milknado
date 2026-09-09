"""Textual panels for the execution TUI — render snapshot data via run_view."""

from __future__ import annotations

from typing import Protocol, cast, final

from rich.console import RenderableType
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp, Resize
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
RUN_COLUMNS = (
    ("Node", 4),
    ("Description", 12),
    ("Status", 12),
    ("Progress", 15),
    ("Elapsed", 7),
)
_COLUMN_INDEX = {"Node": 0, "Description": 1, "Status": 2, "Progress": 3, "Elapsed": 4}
_PRIMARY_COLUMNS = RUN_COLUMNS[:3]
_MIN_FULL_TABLE_WIDTH = sum(width for _, width in RUN_COLUMNS) + 2 * len(RUN_COLUMNS)
_EMPTY_RUN_MESSAGE = (
    "No runs to display.\n\nStart work: milknado run\nRead-only view: milknado watch"
)


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
    #empty { display: none; height: 1fr; margin: 1 2; content-align: center middle; }
    #empty.visible { display: block; }
    #runs { height: 1fr; text-overflow: ellipsis; }
    #runs.hidden { display: none; }
    """

    _runs: tuple[RunSnapshot, ...] = ()
    _selected_run_id: str | None = None
    _cursor_run_id: str | None = None
    _cursor_column = 0
    _column_layout: tuple[tuple[str, int], ...] = ()

    @override
    def compose(self) -> ComposeResult:
        yield Static(id="totals", markup=False)
        yield Static(id="empty", markup=False)
        yield RunTable(id="runs", cursor_type="row")

    def on_mount(self) -> None:
        _ = self._set_columns(self.size.width, preserve_cursor=False)
        self._update_empty_state()

    def on_resize(self, _event: Resize) -> None:
        _ = self.call_after_refresh(self._resize_columns)

    def _resize_columns(self) -> None:
        table = cast(DataTable[RenderableType], self.query_one("#runs", DataTable))
        if self._set_columns(table.scrollable_content_region.width):
            self._render_table()

    def _set_columns(self, width: int, *, preserve_cursor: bool = True) -> bool:
        columns = _PRIMARY_COLUMNS if width < _MIN_FULL_TABLE_WIDTH else RUN_COLUMNS
        fixed_width = sum(
            column_width for label, column_width in columns if label != "Description"
        )
        description_width = max(
            RUN_COLUMNS[1][1],
            width - fixed_width - 2 * len(columns),
        )
        layout = tuple(
            (label, description_width if label == "Description" else column_width)
            for label, column_width in columns
        )
        if layout == self._column_layout:
            return False

        table = cast(DataTable[RenderableType], self.query_one("#runs", DataTable))
        if preserve_cursor and self._runs and table.is_valid_row_index(table.cursor_row):
            self._cursor_run_id = self._runs[table.cursor_row].run_id
            self._cursor_column = table.cursor_column
        _ = table.clear(columns=True)
        for label, column_width in layout:
            _ = table.add_column(label, width=column_width, key=label.lower())
        self._column_layout = layout
        return True

    def _render_table(self) -> None:
        table = cast(DataTable[RenderableType], self.query_one("#runs", DataTable))
        had_focus = table.has_focus
        cursor_column = self._cursor_column if self._cursor_run_id else table.cursor_column
        _ = table.clear(columns=False)
        visible_indexes = tuple(_COLUMN_INDEX[label] for label, _ in self._column_layout)
        for run in self._runs:
            cells = run_row(run)
            description = Text(cells[1], overflow="ellipsis", no_wrap=True)
            visible_cells = (cells[0], description, *cells[2:])
            _ = table.add_row(
                *(visible_cells[index] for index in visible_indexes),
                key=run.run_id,
            )
        if self._runs:
            _ = table.move_cursor(
                row=run_index(self._runs, self._cursor_run_id or self._selected_run_id),
                column=min(cursor_column, len(self._column_layout) - 1),
                animate=False,
            )
        if had_focus:
            _ = table.focus()
        self._cursor_run_id = None

    def _update_empty_state(self) -> None:
        empty = self.query_one("#empty", Static)
        table = cast(DataTable[RenderableType], self.query_one("#runs", DataTable))
        empty.update(_EMPTY_RUN_MESSAGE)
        if self._runs:
            _ = empty.remove_class("visible")
            _ = table.remove_class("hidden")
        else:
            _ = empty.add_class("visible")
            _ = table.add_class("hidden")

    def update(
        self,
        snapshot: ExecutionSnapshot,
        runs: tuple[RunSnapshot, ...],
        selected_run_id: str | None,
    ) -> None:
        self.query_one("#totals", Static).update(subtitle_text(snapshot))
        table = cast(DataTable[RenderableType], self.query_one("#runs", DataTable))
        cursor_run_id = None
        if selected_run_id == self._selected_run_id and table.is_valid_row_index(table.cursor_row):
            cursor_run_id = self._runs[table.cursor_row].run_id
            self._cursor_column = table.cursor_column
        self._cursor_run_id = (
            cursor_run_id if any(run.run_id == cursor_run_id for run in runs) else selected_run_id
        )
        self._runs = runs
        self._selected_run_id = selected_run_id
        _ = self._set_columns(
            table.scrollable_content_region.width or self.size.width, preserve_cursor=False
        )
        self._render_table()
        self._update_empty_state()
        _ = self.call_after_refresh(self._resize_columns)


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
