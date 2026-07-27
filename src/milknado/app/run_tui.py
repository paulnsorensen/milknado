"""Textual operator presentation for immutable execution snapshots."""

from __future__ import annotations

from pathlib import Path
from threading import get_ident
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp, Resize
from textual.widgets import DataTable, Footer, Header, Input, Static

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionController,
    ExecutionSnapshot,
    TerminalRunSnapshot,
)
from milknado.domains.execution import RunLoopResult

if TYPE_CHECKING:
    from textual.worker import Worker

WIDE_MIN_COLUMNS = 100
RunSnapshot = ActiveRunSnapshot | TerminalRunSnapshot


class DetailPane(VerticalScroll):
    """Detail pane that pauses output following before consuming wheel input."""

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self.app._pause_auto_follow()

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self.app._pause_auto_follow()


class RunTable(DataTable):
    """Run selector that preserves the focused table across compact resizes."""

    def on_focus(self) -> None:
        if not self.app.compact:
            self.app.route = "list"


class ExecutionApp(App[RunLoopResult | None]):
    """Responsive, controller-only operator view for a single execution."""

    CSS = """
    #workspace { height: 1fr; }
    #runs { width: 38; min-width: 30; height: 1fr; }
    #detail { width: 1fr; height: 1fr; }
    #output, #events, #actions, #help, #confirmation { margin: 0 1; }
    #output { height: 1fr; overflow-y: auto; border: round $primary; }
    #events { height: 5; border: round $secondary; }
    #guidance { margin: 0 1 1 1; }
    #confirmation { display: none; color: $warning; }
    #help { display: none; }
    .visible { display: block; }
    .compact #workspace { display: block; }
    .compact #runs { width: 1fr; }
    .compact.list #detail { display: none; }
    .compact.detail #runs { display: none; }
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
            yield RunTable(id="runs", cursor_type="row")
            with DetailPane(id="detail"):
                yield Static(id="summary", markup=False)
                with VerticalScroll(id="output"):
                    yield Static(id="output-text", markup=False)
                yield Static(id="events", markup=False)
                yield Static(id="actions", markup=False)
                yield Static(id="help", markup=False)
                yield Static(id="confirmation", markup=False)
                yield Input(placeholder="Queue guidance for the selected run", id="guidance")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.add_columns("Run", "Status", "Progress")
        self._ui_thread_id = get_ident()
        self._unsubscribe = self.controller.subscribe(self._receive_snapshot)
        self._set_layout(self.size.width < WIDE_MIN_COLUMNS)
        if self.feature_branch is not None:
            self._execution_worker = self._run_execution()

    def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def on_resize(self, event: Resize) -> None:
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

    def _refresh_view(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.clear(columns=False)
        for run in self._runs():
            progress = run.progress if isinstance(run, ActiveRunSnapshot) else None
            table.add_row(run.run_id, run.status.value, progress or "—", key=run.run_id)
        if self.selected_run_id is not None:
            table.move_cursor(row=self._run_index())
        selected = self._selected_run()
        self.query_one("#summary", Static).update(self._summary(selected))
        self.query_one("#output-text", Static).update(self._output(selected))
        self.query_one("#events", Static).update(self._events())
        self.query_one("#actions", Static).update(self._actions(selected))
        self.query_one("#help", Static).update(self._help_text(selected))
        active = selected if isinstance(selected, ActiveRunSnapshot) else None
        self.query_one("#guidance", Input).disabled = (
            active is None or not active.actions.can_queue_guidance
        )
        if self.auto_follow:
            self.query_one("#output", VerticalScroll).scroll_end(animate=False)

    def _runs(self) -> tuple[RunSnapshot, ...]:
        return (*self.snapshot.active_runs, *reversed(self.snapshot.terminal_runs))

    def _selected_run(self) -> RunSnapshot | None:
        return next((run for run in self._runs() if run.run_id == self.selected_run_id), None)

    def _selected_active_run(self) -> ActiveRunSnapshot | None:
        run = self._selected_run()
        return run if isinstance(run, ActiveRunSnapshot) else None

    def _run_index(self) -> int:
        return next(
            (
                index
                for index, run in enumerate(self._runs())
                if run.run_id == self.selected_run_id
            ),
            0,
        )

    def _summary(self, run: RunSnapshot | None) -> str:
        totals = (
            f"{self.snapshot.completed} complete · {self.snapshot.failed} failed · "
            f"{self.snapshot.stopped} stopped"
        )
        if run is None:
            return f"{self.snapshot.goal} · {totals}\nNo runs."
        guidance = ", ".join(run.pending_guidance) or "none"
        if isinstance(run, ActiveRunSnapshot):
            stopping = " (stopping)" if run.stop_requested else ""
            status = f"{run.status.value}{stopping} · {run.progress or 'progress unavailable'}"
            guidance_label = "Pending guidance"
        else:
            status = run.status.value
            guidance_label = "Undelivered guidance"
        return (
            f"{self.snapshot.goal} · {totals} · node {run.node_id}\n{run.description}\n"
            f"{status}\n{guidance_label}: {guidance}"
        )

    def _output(self, run: RunSnapshot | None) -> str:
        lines = "\n".join(run.output) if run and run.output else "No output yet."
        suffix = "following newest output" if self.auto_follow else "paused; press r to resume"
        return f"Output ({suffix})\n{lines}"

    def _events(self) -> str:
        events = "\n".join(self.snapshot.event_lines) or "No events yet."
        return f"Events\n{events}"

    def _actions(self, run: RunSnapshot | None) -> str:
        if run is None:
            return "Actions\nNo run selected."
        if not isinstance(run, ActiveRunSnapshot):
            return "Actions\nRun has stopped; output retained for inspection."
        actions = run.actions
        force = actions.force_stop_reason or "available"
        cancel = actions.cancel_reason or "available"
        guidance = actions.guidance_reason or "available"
        return f"Actions\n[c] Cancel: {cancel}\n[f] Force stop: {force}\n[g] Guidance: {guidance}"

    def _help_text(self, run: RunSnapshot | None) -> str:
        actions = ["↑/↓ select", "h close help", "q quit"]
        if self.compact:
            actions.append("enter open" if self.route == "list" else "escape back")
        if isinstance(run, ActiveRunSnapshot):
            if run.actions.can_queue_guidance:
                actions.append("g queue guidance")
            if run.actions.can_cancel:
                actions.append("c cancel")
            if run.actions.can_force_stop:
                actions.append("f force stop")
        if not self.auto_follow:
            actions.append("r resume output")
        return "Help\n" + " · ".join(actions)

    def _set_confirmation(self, action: str, run_id: str | None) -> None:
        self._confirmation = (action, run_id)
        count = len(self.snapshot.active_runs)
        run_label = "active run" if count == 1 else "active runs"
        message = (
            f"Force stop {run_id}? [y] confirm [n] cancel"
            if action == "force"
            else (
                f"Stop scheduling and gracefully stop {count} {run_label}? [y] confirm [n] cancel"
            )
        )
        confirmation = self.query_one("#confirmation", Static)
        confirmation.update(message)
        confirmation.add_class("visible")
        self.set_focus(None)

    def _clear_confirmation(self) -> None:
        self._confirmation = None
        self.query_one("#confirmation", Static).remove_class("visible")

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

    @work(thread=True, group="execution", exclusive=True)
    def _run_execution(self) -> None:
        result = self.controller.run(
            feature_branch=self.feature_branch or "",
            strict=self.strict,
            spec_text=self.spec_text,
            spec_path=self.spec_path,
        )
        self.call_from_thread(self.exit, result)

    @work(thread=True, exclusive=False)
    def _queue_guidance(self, run_id: str, text: str) -> None:
        try:
            accepted = self.controller.queue_guidance(run_id, text)
        except RuntimeError:
            accepted = False
        if accepted:
            self.call_from_thread(self._clear_guidance_if_unchanged, text)
        else:
            self.call_from_thread(
                self.notify, "This run no longer accepts guidance.", severity="warning"
            )

    def _clear_guidance_if_unchanged(self, submitted: str) -> None:
        guidance = self.query_one("#guidance", Input)
        if guidance.value.strip() == submitted:
            guidance.value = ""

    @work(thread=True, group="controls", exclusive=False)
    def _cancel(self, run_id: str) -> None:
        try:
            self.controller.cancel(run_id)
        except RuntimeError as error:
            if str(error) != "execution has finished":
                raise

    @work(thread=True, group="controls", exclusive=False)
    def _force_stop(self, run_id: str) -> None:
        try:
            completed = self.controller.force_stop(run_id)
        except RuntimeError as error:
            if str(error) != "execution has finished":
                raise
            return
        if not completed:
            self.call_from_thread(
                self.notify, "Force-stop cleanup did not finish in time.", severity="warning"
            )

    @work(thread=True, group="controls", exclusive=False)
    def _stop_scheduling(self) -> None:
        try:
            self.controller.stop_scheduling()
        except RuntimeError as error:
            if str(error) != "execution has finished":
                raise

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
