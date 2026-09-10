"""Ordered native session input and per-run editor state."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from textual import on, work
from textual.widget import Widget
from textual.widgets import Button, Input, Select

from milknado.app.run_view import session_view
from milknado.app.session_changes import SessionChangesMixin
from milknado.app.session_navigation import RunSnapshot
from milknado.domains.common import SessionAction, SessionInput


class _SessionController(Protocol):
    def session_input(self, run_id: str, command: SessionInput) -> bool: ...


@dataclass(frozen=True, slots=True)
class _QueuedSessionInput:
    run_id: str
    command: SessionInput
    draft_revision: int
    submission_id: int


_WidgetT = TypeVar("_WidgetT", bound=Widget)


class _SessionHost(Protocol):
    selected_run_id: str | None
    read_only: bool
    is_mounted: bool
    controller: _SessionController

    def selected_run(self) -> RunSnapshot | None: ...

    def query_one(self, _selector: str, _expect_type: type[_WidgetT], /) -> _WidgetT: ...

    def call_from_thread(self, callback: Callable[..., object], *args: object) -> object: ...

    def notify(self, message: str, **kwargs: object) -> object: ...


class SessionCommandsMixin(SessionChangesMixin):
    """Serialize controller calls without blocking Textual's UI loop."""

    read_only: bool = False
    _session_drafts: dict[str, str] = {}
    _session_draft_revisions: dict[str, int] = {}
    _session_submission_ids: dict[str, int] = {}
    _session_actions: dict[str, SessionAction] = {}
    _session_permissions: dict[str, str] = {}
    _input_queue: deque[_QueuedSessionInput] = deque()
    _input_busy: bool = False

    def _session_host(self) -> _SessionHost:
        return cast(_SessionHost, cast(object, self))

    def _init_session_ui(self, *, read_only: bool) -> None:
        self.read_only = read_only
        self._session_drafts = {}
        self._session_draft_revisions = {}
        self._session_submission_ids = {}
        self._session_actions = {}
        self._session_permissions = {}
        self._input_queue = deque()
        self._input_busy = False
        self._init_changes()

    @property
    def session_draft(self) -> str:
        run_id = self._session_host().selected_run_id
        return self._session_drafts.get(run_id, "") if run_id else ""

    @property
    def session_action(self) -> SessionAction | None:
        run_id = self._session_host().selected_run_id
        return self._session_actions.get(run_id) if run_id else None

    @property
    def session_permission(self) -> str:
        run_id = self._session_host().selected_run_id
        return self._session_permissions.get(run_id, "") if run_id else ""

    @on(Input.Changed, "#session-input")
    def preserve_session_draft(self, event: Input.Changed) -> None:
        host = self._session_host()
        if host.selected_run_id and not host.read_only:
            run_id = host.selected_run_id
            self._session_drafts[run_id] = event.value
            self._session_draft_revisions[run_id] = (
                self._session_draft_revisions.get(run_id, 0) + 1
            )

    @on(Select.Changed, "#session-action")
    def select_session_action(self, event: Select.Changed) -> None:
        run_id = self._session_host().selected_run_id
        value = event.value
        if run_id and isinstance(value, str):
            self._session_actions[run_id] = cast(SessionAction, value)

    @on(Select.Changed, "#session-permission")
    def select_session_permission(self, event: Select.Changed) -> None:
        run_id = self._session_host().selected_run_id
        value = event.value
        if run_id and isinstance(value, str):
            self._session_permissions[run_id] = value

    @on(Input.Submitted, "#session-input")
    def submit_session_input(self, _event: Input.Submitted) -> None:
        self._send_session_input()

    @on(Button.Pressed, "#session-submit")
    def click_session_submit(self, _event: Button.Pressed) -> None:
        self._send_session_input()

    def _send_session_input(self) -> None:
        host = self._session_host()
        if host.read_only:
            return
        selected = host.selected_run()
        if selected is None:
            return
        session = session_view(selected)
        if not session.active:
            return
        actions = session.actions
        action = self._selected_action(actions)
        if action is None:
            return
        draft = self.session_draft
        request_id = self._selected_permission() if action in {"approve", "deny"} else ""
        if action in {"steer", "follow_up"} and not draft.strip():
            _ = host.notify("Enter a message for this session action.", severity="warning")
            return
        if action in {"approve", "deny"} and not request_id:
            _ = host.notify("Select the exact permission request first.", severity="warning")
            return
        text = draft if action != "interrupt" else ""
        command = SessionInput(action=action, text=text, request_id=request_id)
        run_id = selected.run_id
        submission_id = self._session_submission_ids.get(run_id, 0) + 1
        self._session_submission_ids[run_id] = submission_id
        self._input_queue.append(
            _QueuedSessionInput(
                run_id=run_id,
                command=command,
                draft_revision=self._session_draft_revisions.get(run_id, 0),
                submission_id=submission_id,
            )
        )
        self._dispatch_session_input()

    def _selected_action(self, actions: tuple[SessionAction, ...]) -> SessionAction | None:
        action_select = cast(
            Select[SessionAction],
            self._session_host().query_one("#session-action", Select),
        )
        value = action_select.value
        return value if isinstance(value, str) and value in actions else None

    def _selected_permission(self) -> str:
        permission_select = cast(
            Select[str],
            self._session_host().query_one("#session-permission", Select),
        )
        value = permission_select.value
        return value if isinstance(value, str) else ""

    def _dispatch_session_input(self) -> None:
        if self._input_busy or not self._input_queue:
            return
        self._input_busy = True
        submission = self._input_queue.popleft()
        _ = self._submit_session_input(submission)

    @work(thread=True, group="controls", exclusive=False)
    def _submit_session_input(self, submission: _QueuedSessionInput) -> None:
        try:
            accepted = self._session_host().controller.session_input(
                submission.run_id, submission.command
            )
            error = None
        except (RuntimeError, ValueError) as exc:
            accepted = False
            error = str(exc)
        _ = self._session_host().call_from_thread(
            self._session_input_result, submission, accepted, error
        )

    def _session_input_result(
        self, submission: _QueuedSessionInput, accepted: bool, error: str | None
    ) -> None:
        self._input_busy = False
        host = self._session_host()
        if accepted:
            owns_latest_draft = (
                submission.command.action != "interrupt"
                and self._session_draft_revisions.get(submission.run_id, 0)
                == submission.draft_revision
                and self._session_submission_ids.get(submission.run_id) == submission.submission_id
            )
            if owns_latest_draft:
                self._session_drafts[submission.run_id] = ""
                if host.selected_run_id == submission.run_id:
                    host.query_one("#session-input", Input).value = ""
            _ = host.notify("Session input queued.")
        else:
            detail = f": {error}" if error else ""
            _ = host.notify(f"Session input was rejected{detail}", severity="warning")
        self._dispatch_session_input()
