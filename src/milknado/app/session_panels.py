"""Native Textual panes for structured session, change, and detail views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast, final

from rich.console import RenderableType
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp
from textual.widgets import Button, DataTable, Input, Select, Static
from typing_extensions import override

from milknado.adapters import ChangedFile
from milknado.app.graph_view import detail_navigation_text, node_inspector_text
from milknado.app.session_view import (
    action_options,
    error_text,
    permission_options,
    session_state_text,
    transcript_text,
)
from milknado.domains.common import MikadoNode, SessionAction, SessionView
from milknado.domains.graph import NodeDetailSnapshot


class _OutputHost(Protocol):
    def pause_auto_follow(self) -> None: ...


@final
class OutputPanel(VerticalScroll):
    """Suspend following when the user scrolls transcript output."""

    def on_mouse_scroll_up(self, _event: MouseScrollUp) -> None:
        cast(_OutputHost, cast(object, self.app)).pause_auto_follow()

    def on_mouse_scroll_down(self, _event: MouseScrollDown) -> None:
        cast(_OutputHost, cast(object, self.app)).pause_auto_follow()


@dataclass(frozen=True, slots=True)
class SessionPanelState:
    view: SessionView
    run_id: str | None = None
    read_only: bool = False
    draft: str = ""
    legacy_guidance: bool = False
    action: SessionAction | None = None
    permission_id: str = ""


@dataclass(frozen=True, slots=True)
class ChangesPanelState:
    files: tuple[ChangedFile, ...] = ()
    selected_path: str | None = None
    diff: str = ""
    error: str | None = None
    loading: bool = False


@final
class SessionPanel(VerticalScroll):
    """Transcript and explicit protocol controls for one selected run."""

    _run_id: str | None = None
    _action_choices: tuple[tuple[str, SessionAction], ...] = ()
    _permission_choices: tuple[tuple[str, str], ...] = ()

    DEFAULT_CSS = """
    SessionPanel { height: 1fr; }
    #session-state { height: auto; margin: 0 1; }
    #session-state.hidden { display: none; }
    #output { height: 1fr; min-height: 3; margin: 0 1; border: round $primary; }
    #session-errors { height: auto; max-height: 5; margin: 0 1; border: round $error; }
    #session-errors.hidden { display: none; }
    #structured-controls { height: auto; }
    #session-controls, #session-input-row { height: auto; margin: 0 1; }
    #session-action, #session-permission { width: 1fr; }
    #session-input { width: 1fr; }
    #session-submit { width: auto; min-width: 8; }
    #structured-controls.hidden, #session-input-row.hidden { display: none; }
    #guidance { margin: 0 1 1 1; }
    #actions { height: auto; margin: 0 1; }
    """

    @override
    def compose(self) -> ComposeResult:
        yield Static(id="session-state", markup=False)
        with OutputPanel(id="output") as output:
            output.border_title = "Session"
            yield Static(id="output-text", markup=False)
        with VerticalScroll(id="session-errors") as errors:
            errors.border_title = "Errors"
            yield Static(id="session-errors-text", markup=False)
        with Vertical(id="structured-controls"):
            with Horizontal(id="session-controls"):
                yield Select([], prompt="Action", id="session-action")
                yield Select([], prompt="Permission request", id="session-permission")
            with Horizontal(id="session-input-row"):
                yield Input(placeholder="Message the selected session", id="session-input")
                yield Button("Send", id="session-submit", variant="primary")
        yield Static(id="actions", markup=False)
        yield Input(placeholder="Queue guidance for the selected run", id="guidance")

    def update(self, state: SessionPanelState) -> None:
        view = state.view
        label = f"Session {session_state_text(view)}"
        if view.context:
            label += f" · {view.context.family}"
        session_state = self.query_one("#session-state", Static)
        session_state.update(label)
        _ = session_state.set_class(view.context is None, "hidden")
        self.query_one("#output-text", Static).update(transcript_text(view))
        self._update_errors(view)
        self.query_one("#actions", Static).display = not state.read_only and view.context is None
        structured = bool(state.run_id and view.active and view.actions and not state.read_only)
        selection_changed = state.run_id != self._run_id
        with self.prevent(Input.Changed, Select.Changed):
            self._update_selectors(state, structured, selection_changed)
            self._update_inputs(state, structured, selection_changed)
        self._run_id = state.run_id

    def _update_errors(self, view: SessionView) -> None:
        error_box = self.query_one("#session-errors", VerticalScroll)
        error_box.query_one("#session-errors-text", Static).update(error_text(view))
        has_errors = any(event.kind == "error" for event in view.events)
        _ = error_box.set_class(not has_errors, "hidden")

    def _update_selectors(
        self, state: SessionPanelState, structured: bool, selection_changed: bool
    ) -> None:
        view = state.view
        permissions = permission_options(view) if structured else ()
        available: tuple[SessionAction, ...] = tuple(
            action
            for action in view.actions
            if structured and (action not in ("approve", "deny") or permissions)
        )
        controls = self.query_one("#structured-controls", Vertical)
        _ = controls.set_class(not structured, "hidden")
        action_select = cast(Select[SessionAction], self.query_one("#session-action", Select))
        choices = action_options(available)
        if choices != self._action_choices:
            action_select.set_options(choices)
            self._action_choices = choices
        action = (
            state.action
            if state.action in available
            else (available[0] if available else Select.NULL)
        )
        if action_select.value != action and (
            selection_changed
            or not action_select.has_focus
            or action_select.value not in available
        ):
            action_select.value = action
        permission_select = cast(Select[str], self.query_one("#session-permission", Select))
        if permissions != self._permission_choices:
            permission_select.set_options(permissions)
            self._permission_choices = permissions
        permission = (
            state.permission_id
            if any(value == state.permission_id for _, value in permissions)
            else Select.NULL
        )
        if permission_select.value != permission:
            permission_select.value = permission
        permission_select.display = bool(permissions)

    def _update_inputs(
        self, state: SessionPanelState, structured: bool, selection_changed: bool
    ) -> None:
        input_row = self.query_one("#session-input-row", Horizontal)
        input_row.display = structured
        message_input = self.query_one("#session-input", Input)
        if message_input.value != state.draft and (
            selection_changed or not message_input.has_focus
        ):
            message_input.value = state.draft
        message_input.disabled = not structured
        self.query_one("#session-submit", Button).disabled = not structured
        guidance = self.query_one("#guidance", Input)
        guidance.display = state.legacy_guidance and not state.read_only
        guidance.disabled = not state.legacy_guidance or state.read_only

    def show_legacy_output(self, output: str) -> None:
        """Keep legacy worker output visible when no structured events exist."""
        _ = self.query_one("#output-text", Static).update(output or "No output yet.")


@final
class ChangesPanel(Vertical):
    """Changed-file navigator and bounded unified diff viewer."""

    _rendered_files: tuple[ChangedFile, ...] = ()

    DEFAULT_CSS = """
    ChangesPanel { height: 1fr; }
    #changes-state { height: auto; margin: 0 1; }
    #changes-files { height: 1fr; min-height: 4; margin: 0 1; }
    #diff { height: 1fr; margin: 0 1; border: round $secondary; }
    #changes-files.hidden { display: none; }
    """

    @override
    def compose(self) -> ComposeResult:
        yield Static(id="changes-state", markup=False)
        yield DataTable[RenderableType](id="changes-files", cursor_type="row")
        with VerticalScroll(id="diff") as diff:
            diff.border_title = "Unified diff"
            yield Static(id="diff-text", markup=False)

    def on_mount(self) -> None:
        table = cast(DataTable[RenderableType], self.query_one("#changes-files", DataTable))
        _ = table.add_column("Status", width=8, key="status")
        _ = table.add_column("Path", key="path")
        _ = table.add_column("+/-", width=8, key="counts")

    def update(self, state: ChangesPanelState) -> None:
        tables = self.query("#changes-files")
        if not tables:
            # Mid-teardown: child already unmounted while a worker callback fires.
            return
        table = cast(DataTable[RenderableType], tables.first())
        self._update_files(table, state)
        if state.files:
            _ = table.remove_class("hidden")
            message = f"{len(state.files)} changed file{'s' if len(state.files) != 1 else ''}"
            self.query_one("#changes-state", Static).update(message)
        else:
            _ = table.add_class("hidden")
            message = "Loading changes..." if state.loading else "No changed files."
            self.query_one("#changes-state", Static).update(state.error or message)
        self.query_one("#diff-text", Static).update(
            state.diff or state.error or "Select a changed file."
        )

    def _update_files(self, table: DataTable[RenderableType], state: ChangesPanelState) -> None:
        if state.files == self._rendered_files:
            return
        cursor_path = state.selected_path
        if table.has_focus and 0 <= table.cursor_row < len(self._rendered_files):
            cursor_path = self._rendered_files[table.cursor_row].path
        self._rendered_files = state.files
        _ = table.clear()
        for changed in state.files:
            _ = table.add_row(
                changed.status, changed.path, _line_counts(changed), key=changed.path
            )
        if state.files:
            paths = tuple(item.path for item in state.files)
            selected = paths.index(state.selected_path) if state.selected_path in paths else 0
            row = paths.index(cursor_path) if cursor_path in paths else selected
            table.move_cursor(row=row, animate=False)


def _line_counts(changed: ChangedFile) -> str:
    if changed.added is None or changed.removed is None:
        return "binary"
    return f"+{changed.added}/-{changed.removed}"


@final
class DetailsPanel(VerticalScroll):
    """Full node brief and immutable run metadata."""

    DEFAULT_CSS = """
    DetailsPanel { height: 1fr; padding: 0 1; }
    #brief { height: auto; margin: 0 0 1 0; }
    #metadata { height: auto; }
    #detail-navigation { height: auto; margin: 0 0 1 0; color: $text-muted; }
    """

    @override
    def compose(self) -> ComposeResult:
        yield Static(id="brief", markup=False)
        yield Static(id="detail-navigation", markup=False)
        yield Static(id="metadata", markup=False)

    def update(
        self,
        brief: str,
        metadata: str,
        *,
        node: MikadoNode | None = None,
        detail: NodeDetailSnapshot | None = None,
    ) -> None:
        offset = self.scroll_offset
        self.query_one("#brief", Static).update(
            node_inspector_text(node, detail) if node is not None else brief
        )
        self.query_one("#detail-navigation", Static).update(
            detail_navigation_text(detail) if node is not None else ""
        )
        self.query_one("#metadata", Static).update("" if node is not None else metadata)
        _ = self.call_after_refresh(self.scroll_to, x=offset.x, y=offset.y, animate=False)
