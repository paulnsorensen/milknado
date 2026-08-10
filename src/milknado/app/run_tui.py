"""Textual operator presentation for immutable execution snapshots."""

from __future__ import annotations

from pathlib import Path
from threading import get_ident
from typing import TYPE_CHECKING

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Input, Static

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionController,
    ExecutionSnapshot,
    TerminalRunSnapshot,
)
from milknado.app.run_commands import ExecutionCommandsMixin
from milknado.app.run_panels import RunDetailPanel, RunListPanel
from milknado.app.run_view import confirmation_text, events_text, subtitle_text
from milknado.domains.execution import RunLoopResult

if TYPE_CHECKING:
    from textual.worker import Worker

# Two-pane needs the full run table (64) plus a readable detail pane. Below this
# the compact route shows the same table full-width, so nothing is lost.
WIDE_MIN_COLUMNS = 116
RunSnapshot = ActiveRunSnapshot | TerminalRunSnapshot


class ExecutionApp(ExecutionCommandsMixin, App[RunLoopResult | None]):
    """Responsive, controller-only operator view for a single execution."""

    CSS = """
    #workspace { height: 1fr; }
    #events { height: auto; max-height: 8; margin: 0 1; border: round $secondary; }
    .visible { display: block; }
    .compact #workspace { display: block; }
    .compact #totals { display: block; }
    .compact #run-panel { width: 1fr; }
    .compact.list #detail { display: none; }
    .compact.detail #run-panel { display: none; }
    """
    BINDINGS = [
        ("up", "previous_run", "Previous run"),
        ("down", "next_run", "Next run"),
        ("enter", "open_detail", "Open selected run"),
        ("escape", "back", "Back"),
        ("g", "focus_guidance", "Queue guidance"),
        ("c", "cancel", "Cancel"),
        ("f", "force", "Force stop"),
        ("r", "resume_output", "Resume output"),
        ("h", "help", "Help"),
        ("q", "quit_all", "Quit"),
    ]

    def __init__(
        self,
        controller: ExecutionController,
        *,
        feature_branch: str | None = None,
        strict: bool = False,
        spec_text: str | None = None,
        spec_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.snapshot = controller.snapshot()
        self.feature_branch = feature_branch
        self.strict = strict
        self.spec_text = spec_text
        self.spec_path = spec_path
        runs = self._runs()
        self.selected_run_id = runs[0].run_id if runs else None
        self.route = "list"
        self.compact = False
        self.auto_follow = True
        self._ui_thread_id: int | None = None
        self._execution_worker: Worker | None = None
        self._unsubscribe = None
        self._confirmation: tuple[str, str | None] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            yield RunListPanel(id="run-panel")
            yield RunDetailPanel(id="detail")
        # Events describe the execution, not the selected run, so they stay visible
        # across both compact routes instead of hiding inside the detail pane.
        yield Static(id="events", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self._ui_thread_id = get_ident()
        self._unsubscribe = self.controller.subscribe(self._receive_snapshot)
        self._set_layout(self.size.width < WIDE_MIN_COLUMNS)
        if self.feature_branch is not None:
            self._execution_worker = self._run_execution()

    def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def on_resize(self, event) -> None:
        self._set_layout(event.size.width < WIDE_MIN_COLUMNS)

    def _receive_snapshot(self, snapshot: ExecutionSnapshot) -> None:
        """Marshal a controller-thread replacement snapshot onto Textual's loop."""
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
        self.set_class(compact, "compact")
        self.set_class(compact and self.route == "list", "list")
        self.set_class(compact and self.route == "detail", "detail")
        if not self.auto_follow:
            self.query_one("#detail", RunDetailPanel).preserve_output_offset()

    def _refresh_view(self) -> None:
        self.title = self.snapshot.goal
        self.sub_title = subtitle_text(self.snapshot)
        self.query_one("#run-panel", RunListPanel).update(
            self.snapshot, self._runs(), self.selected_run_id
        )
        self.query_one("#detail", RunDetailPanel).update(
            self.snapshot,
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

    def _set_confirmation(self, action: str, run_id: str | None) -> None:
        self._confirmation = (action, run_id)
        message = confirmation_text(action, run_id, len(self.snapshot.active_runs))
        self.query_one("#detail", RunDetailPanel).set_confirmation(message)
        self.set_focus(None)

    def _clear_confirmation(self) -> None:
        self._confirmation = None
        self.query_one("#detail", RunDetailPanel).clear_confirmation()

    @on(DataTable.RowSelected, "#runs")
    def select_row(self, event: DataTable.RowSelected) -> None:
        self.selected_run_id = str(event.row_key.value)
        self._refresh_view()
        if self.compact:
            self.route = "detail"
            self._set_layout(True)

    @on(Input.Submitted, "#guidance")
    def submit_guidance(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.set_focus(None)
        run = self._selected_active_run()
        if text and run is not None and run.actions.can_queue_guidance:
            self._queue_guidance(run.run_id, text)

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

    def action_back(self) -> None:
        if self._confirmation is not None:
            self._clear_confirmation()
        elif self.compact and self.route == "detail":
            self.route = "list"
            self._set_layout(True)
        else:
            self.query_one("#help", Static).remove_class("visible")

    def action_focus_guidance(self) -> None:
        run = self._selected_active_run()
        if run is not None and run.actions.can_queue_guidance:
            self.query_one("#guidance", Input).focus()

    def action_cancel(self) -> None:
        run = self._selected_active_run()
        if run is not None and run.actions.can_cancel:
            self._cancel(run.run_id)

    def action_force(self) -> None:
        run = self._selected_active_run()
        if run is not None and run.actions.can_force_stop:
            self._set_confirmation("force", run.run_id)

    def action_resume_output(self) -> None:
        self.auto_follow = True
        self._refresh_view()

    def action_help(self) -> None:
        help_panel = self.query_one("#help", Static)
        help_panel.toggle_class("visible")

    def action_quit_all(self) -> None:
        if self.snapshot.active_runs or self._execution_in_flight():
            self._set_confirmation("quit", None)
        else:
            self.exit()

    def _execution_in_flight(self) -> bool:
        return self._execution_worker is not None and not self._execution_worker.is_finished

    def _pause_auto_follow(self) -> None:
        self.auto_follow = False
        self._refresh_view()

    def on_key(self, event) -> None:
        if self._confirmation is None:
            if event.key in {"home", "end", "pageup", "pagedown"}:
                self._pause_auto_follow()
            return
        if event.key == "y":
            action, run_id = self._confirmation
            self._clear_confirmation()
            if action == "force" and run_id is not None:
                self._force_stop(run_id)
            elif action == "quit":
                self._stop_scheduling()
        elif event.key in {"n", "escape"}:
            self._clear_confirmation()
        event.stop()


def run_execution_tui(
    controller: ExecutionController,
    *,
    feature_branch: str,
    strict: bool = False,
    spec_text: str | None = None,
    spec_path: Path | None = None,
) -> RunLoopResult | None:
    """Run the controller-backed Textual adapter and return its terminal result."""
    return ExecutionApp(
        controller,
        feature_branch=feature_branch,
        strict=strict,
        spec_text=spec_text,
        spec_path=spec_path,
    ).run()
