"""Native Textual panes for structured session, change, and detail views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, final

from rich.console import RenderableType
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Input, Select, Static
from typing_extensions import override

from milknado.adapters import ChangedFile
from milknado.app.session_view import (
    action_options,
    error_text,
    permission_options,
    session_state_text,
    transcript_text,
)
from milknado.domains.common import SessionAction, SessionView


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
        with VerticalScroll(id="output") as output:
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
        with self.prevent(Input.Changed, Select.Changed):
            self._update_selectors(state, structured)
            self._update_inputs(state, structured)

    def _update_errors(self, view: SessionView) -> None:
        error_box = self.query_one("#session-errors", VerticalScroll)
        error_box.query_one("#session-errors-text", Static).update(error_text(view))
        has_errors = any(event.kind == "error" for event in view.events)
        _ = error_box.set_class(not has_errors, "hidden")

    def _update_selectors(self, state: SessionPanelState, structured: bool) -> None:
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
        action_select.value = (
            state.action
            if state.action in available
            else available[0]
            if available
            else Select.NULL
        )
        permission_select = cast(Select[str], self.query_one("#session-permission", Select))
        if permissions != self._permission_choices:
            permission_select.set_options(permissions)
            self._permission_choices = permissions
        permission_select.value = (
            state.permission_id
            if any(value == state.permission_id for _, value in permissions)
            else Select.NULL
        )
        permission_select.display = bool(permissions)

    def _update_inputs(self, state: SessionPanelState, structured: bool) -> None:
        input_row = self.query_one("#session-input-row", Horizontal)
        input_row.display = structured
        message_input = self.query_one("#session-input", Input)
        if message_input.value != state.draft:
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
        table = cast(DataTable[RenderableType], self.query_one("#changes-files", DataTable))
        self._update_files(table, state)
        if state.files:
            _ = table.remove_class("hidden")
            message = f"{len(state.files)} changed file"
            self.query_one("#changes-state", Static).update(
                message + ("" if len(state.files) == 1 else "s")
            )
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
            row = next(
                (index for index, item in enumerate(state.files) if item.path == cursor_path),
                self._selected_index(state),
            )
            table.move_cursor(row=row, animate=False)

    @staticmethod
    def _selected_index(state: ChangesPanelState) -> int:
        return next(
            (index for index, item in enumerate(state.files) if item.path == state.selected_path),
            0,
        )


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
    """

    @override
    def compose(self) -> ComposeResult:
        yield Static(id="brief", markup=False)
        yield Static(id="metadata", markup=False)

    def update(self, brief: str, metadata: str) -> None:
        self.query_one("#brief", Static).update(brief)
        self.query_one("#metadata", Static).update(metadata)
