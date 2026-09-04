"""Textual operator controls for execution snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual import on
from textual.widgets import Input

from milknado.app.run import ExecutionController
from milknado.app.run_commands import ExecutionCommandsMixin
from milknado.app.run_panels import RunDetailPanel
from milknado.app.run_view import confirmation_text
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.domains.execution import RunLoopResult

if TYPE_CHECKING:
    from textual.events import Key
    from textual.worker import Worker


class ExecutionApp(ExecutionCommandsMixin, ExecutionSnapshotApp):
    """Controller-backed operator view for one execution."""

    BINDINGS = [  # noqa: V107 - Textual reads binding configuration
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
        self.controller = controller
        self.feature_branch = feature_branch
        self.strict = strict
        self.spec_text = spec_text
        self.spec_path = spec_path
        self._execution_worker: Worker | None = None
        self._confirmation: tuple[str, str | None] | None = None
        super().__init__(controller)

    def on_mount(self) -> None:
        super().on_mount()
        if self.feature_branch is not None:
            self._execution_worker = self._run_execution()

    @on(Input.Submitted, "#guidance")
    def submit_guidance(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.set_focus(None)
        run = self._selected_active_run()
        if text and run is not None and run.actions.can_queue_guidance:
            self._queue_guidance(run.run_id, text)

    def _set_confirmation(self, action: str, run_id: str | None) -> None:
        self._confirmation = (action, run_id)
        message = confirmation_text(action, run_id, len(self.snapshot.active_runs))
        self.query_one("#detail", RunDetailPanel).set_confirmation(message)
        self.set_focus(None)

    def _clear_confirmation(self) -> None:
        self._confirmation = None
        self.query_one("#detail", RunDetailPanel).clear_confirmation()

    def action_back(self) -> None:
        if self._confirmation is not None:
            self._clear_confirmation()
        else:
            super().action_back()

    def action_focus_guidance(self) -> None:  # noqa: V105 - Textual binding action
        run = self._selected_active_run()
        if run is not None and run.actions.can_queue_guidance:
            self.query_one("#guidance", Input).focus()

    def action_cancel(self) -> None:  # noqa: V105 - Textual binding action
        run = self._selected_active_run()
        if run is not None and run.actions.can_cancel:
            self._cancel(run.run_id)

    def action_force(self) -> None:  # noqa: V105 - Textual binding action
        run = self._selected_active_run()
        if run is not None and run.actions.can_force_stop:
            self._set_confirmation("force", run.run_id)

    def action_quit_all(self) -> None:  # noqa: V105 - Textual binding action
        if self.snapshot.active_runs or self._execution_in_flight():
            self._set_confirmation("quit", None)
        else:
            self.exit()

    def _execution_in_flight(self) -> bool:
        return self._execution_worker is not None and not self._execution_worker.is_finished

    def on_key(self, event: Key) -> None:
        if self._confirmation is None:
            super().on_key(event)
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
