"""Textual operator controls for execution snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from textual import on
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.screen import ModalScreen
from textual.widgets import Input, Static
from typing_extensions import override

from milknado.app.run import ExecutionController, ExecutionSnapshot
from milknado.app.run_commands import ExecutionCommandsMixin
from milknado.app.run_view import confirmation_text, session_view
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.domains.execution import RunLoopResult

__all__ = ["ExecutionApp", "run_execution_tui"]

if TYPE_CHECKING:
    from textual.widget import Widget
    from textual.worker import Worker


class _ConfirmationOverlay(ModalScreen[bool]):
    """Keep destructive consent separate from the underlying editor."""

    SCOPED_CSS: ClassVar[bool] = False  # noqa: V107 - Textual omits private class selectors
    BINDINGS: ClassVar[list[BindingType]] = [  # noqa: V107 - Textual reads bindings
        ("y", "confirm", "Confirm"),
        ("n", "cancel", "Cancel"),
        ("escape", "cancel", "Cancel"),
    ]
    DEFAULT_CSS: ClassVar[str] = """
    #confirmation-screen { align: center middle; }
    #confirmation-overlay {
        margin: 0 1;
        width: 1fr;
        height: auto;
        max-height: 100%;
        padding: 1 2;
        border: round $warning;
        background: $surface;
        color: $warning;
        overflow-y: auto;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__(id="confirmation-screen")
        self.message: str = message

    @override
    def compose(self) -> ComposeResult:
        yield Static(self.message, id="confirmation-overlay", classes="visible", markup=False)

    def action_confirm(self) -> None:
        _ = self.dismiss(True)

    def action_cancel(self) -> None:
        _ = self.dismiss(False)


class ExecutionApp(ExecutionCommandsMixin, ExecutionSnapshotApp):
    """Controller-backed operator view for one execution."""

    BINDINGS: ClassVar[list[BindingType]] = [  # noqa: V107 - Textual reads binding configuration
        ("g", "focus_guidance", "Queue guidance"),
        ("c", "cancel", "Cancel"),
        ("f", "force", "Force stop"),
    ]

    def __init__(
        self,
        controller: ExecutionController,
        *,
        feature_branch: str | None = None,
        strict: bool = False,
        spec_text: str | None = None,
        spec_path: Path | None = None,
        allow_protected: bool = False,
    ) -> None:
        self.controller: ExecutionController = controller
        self.feature_branch: str | None = feature_branch
        self.strict: bool = strict
        self.spec_text: str | None = spec_text
        self.spec_path: Path | None = spec_path
        self.allow_protected: bool = allow_protected
        self._execution_worker: Worker[None] | None = None
        self._confirmation: tuple[str, str | None] | None = None
        self._confirmation_run_ids: frozenset[str] = frozenset()
        self._confirmation_focus: Widget | None = None
        super().__init__(controller)

    @override
    def on_mount(self) -> None:
        if self.feature_branch is not None:
            self._execution_worker = self._run_execution()

    @override
    def show_snapshot(self, snapshot: ExecutionSnapshot) -> None:
        super().show_snapshot(snapshot)
        if self._confirmation is not None and not self._confirmation_is_valid(snapshot):
            self._clear_confirmation()

    @on(Input.Submitted, "#guidance")
    def submit_guidance(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.set_focus(None)
        run = self._selected_active_run()
        if text and run is not None and run.actions.can_queue_guidance:
            _ = self._queue_guidance(run.run_id, text)

    def _set_confirmation(self, action: str, run_id: str | None) -> None:
        if self._confirmation is not None:
            return
        self._confirmation_focus = self.screen.focused
        self._confirmation = (action, run_id)
        self._confirmation_run_ids = frozenset(run.run_id for run in self.snapshot.active_runs)
        message = confirmation_text(action, run_id, len(self.snapshot.active_runs))
        _ = self.push_screen(_ConfirmationOverlay(message), self._finish_confirmation)

    def _clear_confirmation(self) -> None:
        self._confirmation = None
        _ = cast(_ConfirmationOverlay, self.screen).dismiss(False)

    def _confirmation_is_valid(self, snapshot: ExecutionSnapshot) -> bool:
        if self._confirmation is None:
            return False
        action, run_id = self._confirmation
        if action == "quit":
            return self._confirmation_run_ids == frozenset(
                run.run_id for run in snapshot.active_runs
            )
        return any(
            run.run_id == run_id and run.actions.can_force_stop for run in snapshot.active_runs
        )

    def _finish_confirmation(self, confirmed: bool | None) -> None:
        request = self._confirmation
        valid = confirmed is True and self._confirmation_is_valid(self.controller.snapshot())
        self._confirmation = None
        focus = self._confirmation_focus
        self._confirmation_focus = None
        if isinstance(focus, Input) and self.compact and not focus.disabled:
            self.action_open_detail()
        _ = self.call_after_refresh(self._restore_focus, focus)
        if valid and request is not None:
            action, run_id = request
            if action == "force" and run_id is not None:
                _ = self._force_stop(run_id)
            elif action == "quit":
                _ = self._stop_scheduling()

    @override
    async def action_back(self) -> None:
        if self._confirmation is not None:
            self._clear_confirmation()
        else:
            await super().action_back()

    def action_focus_guidance(self) -> None:  # noqa: V105 - Textual binding action
        selected = self.selected_run()
        if session_view(selected).actions:
            self.action_focus_session()
            return
        run = self._selected_active_run()
        if run is not None and run.actions.can_queue_guidance:
            self.action_open_detail()
            _ = self.call_after_refresh(self.query_one("#guidance", Input).focus)

    def action_cancel(self) -> None:  # noqa: V105 - Textual binding action
        run = self._selected_active_run()
        if run is not None and run.actions.can_cancel:
            _ = self._cancel(run.run_id)

    def action_force(self) -> None:  # noqa: V105 - Textual binding action
        run = self._selected_active_run()
        if run is not None and run.actions.can_force_stop:
            self._set_confirmation("force", run.run_id)

    @override
    def action_quit_all(self) -> None:  # noqa: V105 - Textual binding action
        if self.snapshot.active_runs or self._execution_in_flight():
            self._set_confirmation("quit", None)
        else:
            self.exit()

    def _execution_in_flight(self) -> bool:
        return self._execution_worker is not None and not self._execution_worker.is_finished


def run_execution_tui(
    controller: ExecutionController,
    *,
    feature_branch: str,
    strict: bool = False,
    spec_text: str | None = None,
    spec_path: Path | None = None,
    allow_protected: bool = False,
) -> RunLoopResult | None:
    """Run the controller-backed Textual adapter and return its terminal result."""
    return ExecutionApp(
        controller,
        feature_branch=feature_branch,
        strict=strict,
        spec_text=spec_text,
        spec_path=spec_path,
        allow_protected=allow_protected,
    ).run()
