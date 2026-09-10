"""Textual panels for the execution TUI — render snapshot data via run_view."""

from __future__ import annotations

from os import environ
from typing import Protocol, cast, final

from rich.console import RenderableType
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp, Resize
from textual.widgets import DataTable, Static, TabbedContent, TabPane
from typing_extensions import override

from milknado.app.run import ActiveRunSnapshot, ExecutionSnapshot, TerminalRunSnapshot
from milknado.app.run_view import (
    actions_text,
    details_text,
    output_body,
    output_border_title,
    run_index,
    run_row,
    session_view,
    subtitle_text,
    summary_text,
)
from milknado.app.session_panels import (
    ChangesPanel,
    DetailsPanel,
    SessionPanel,
    SessionPanelState,
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
    RunListPanel { width: 40; height: 1fr; }
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
        cursor_column = self._cursor_column if self._cursor_run_id else table.cursor_column
        _ = table.clear(columns=False)
        visible_indexes = tuple(_COLUMN_INDEX[label] for label, _ in self._column_layout)
        color = "NO_COLOR" not in environ
        for run in self._runs:
            cells = run_row(run, color=color)
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
    """Selected node's Session, Changes, and Details panes."""

    DEFAULT_CSS = """
    RunDetailPanel { width: 1fr; height: 1fr; }
    #summary { height: auto; margin: 0 1; }
    #run-tabs { height: 1fr; }
    """

    @override
    def compose(self) -> ComposeResult:
        yield Static(id="summary", markup=False)
        with TabbedContent(initial="session", id="run-tabs"):
            with TabPane("Session", id="session"):
                yield SessionPanel(id="session-panel")
            with TabPane("Changes", id="changes"):
                yield ChangesPanel(id="changes-panel")
            with TabPane("Details", id="details"):
                yield DetailsPanel(id="details-panel")

    def on_mouse_scroll_up(self, _event: MouseScrollUp) -> None:
        cast(_ExecutionAppLike, cast(object, self.app)).pause_auto_follow()

    def on_mouse_scroll_down(self, _event: MouseScrollDown) -> None:
        cast(_ExecutionAppLike, cast(object, self.app)).pause_auto_follow()

    def update(self, selected: RunSnapshot | None, *, auto_follow: bool) -> None:
        session = session_view(selected)
        app = cast(_ExecutionAppLike, cast(object, self.app))
        self.query_one("#summary", Static).update(summary_text(selected, compact=app.compact))
        self.query_one("#output", VerticalScroll).border_title = output_border_title(
            auto_follow=auto_follow
        )
        session_panel = self.query_one("#session-panel", SessionPanel)
        session_panel.update(
            SessionPanelState(
                view=session,
                run_id=selected.run_id if selected else None,
                read_only=getattr(app, "read_only", False),
                draft=getattr(app, "session_draft", ""),
                legacy_guidance=(
                    isinstance(selected, ActiveRunSnapshot)
                    and not session.actions
                    and not getattr(app, "read_only", False)
                ),
                action=getattr(app, "session_action", None),
                permission_id=getattr(app, "session_permission", ""),
            )
        )
        if selected is not None and not session.events:
            session_panel.show_legacy_output(output_body(selected))
        self.query_one("#actions", Static).update(
            actions_text(selected, None if getattr(app, "read_only", False) else session)
        )
        brief, metadata = details_text(selected)
        self.query_one("#details-panel", DetailsPanel).update(brief, metadata)
        if auto_follow:
            self.query_one("#output", VerticalScroll).scroll_end(animate=False)

    def preserve_output_offset(self) -> None:
        """Re-apply the paused output position after a relayout."""
        output = self.query_one("#output", VerticalScroll)
        position = output.scroll_offset.y
        _ = self.call_after_refresh(output.scroll_to, y=position, animate=False)
