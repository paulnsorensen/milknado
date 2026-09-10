"""Bounded Git inspection for the selected session and file."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Protocol, TypeVar, cast

from textual import on, work
from textual.message_pump import MessagePump
from textual.widget import Widget
from textual.widgets import DataTable

from milknado.adapters import ChangedFile, GitAdapter
from milknado.app.run_view import session_view
from milknado.app.session_navigation import RunSnapshot
from milknado.app.session_panels import ChangesPanel, ChangesPanelState
from milknado.domains.common import GitOperationError, SessionContext

_WidgetT = TypeVar("_WidgetT", bound=Widget)


class _ChangesHost(Protocol):
    is_mounted: bool

    def selected_run(self) -> RunSnapshot | None: ...

    def query_one(self, _selector: str, _expect_type: type[_WidgetT], /) -> _WidgetT: ...

    def call_from_thread(self, callback: Callable[..., object], *args: object) -> object: ...


@dataclass(frozen=True, slots=True)
class _ChangesRequest:
    token: int
    run_id: str
    context: SessionContext


@dataclass(frozen=True, slots=True)
class _DiffRequest:
    token: int
    run_id: str
    context: SessionContext
    path: str


class SessionChangesMixin(metaclass=type(MessagePump)):
    selected_file_path: str | None = None
    _changes_files: tuple[ChangedFile, ...] = ()
    _changes_error: str | None = None
    _changes_diff: str = ""
    _changes_identity: tuple[str, SessionContext] | None = None
    _diff_identity: tuple[str, SessionContext, str] | None = None
    _changes_token: int = 0
    _diff_token: int = 0
    _changes_busy: bool = False
    _diff_busy: bool = False
    _changes_loading: bool = False
    _pending_changes: _ChangesRequest | None = None
    _pending_diff: _DiffRequest | None = None
    _last_changes_request: float = 0.0

    def _changes_host(self) -> _ChangesHost:
        return cast(_ChangesHost, cast(object, self))

    def _init_changes(self) -> None:
        self.selected_file_path = None
        self._changes_files = ()
        self._changes_diff = ""
        self._changes_error = "Changes unavailable: no session worktree."
        self._changes_identity = None
        self._diff_identity = None
        self._changes_token = self._diff_token = 0
        self._changes_busy = self._diff_busy = self._changes_loading = False
        self._pending_changes = None
        self._pending_diff = None
        self._last_changes_request = 0.0

    def refresh_session_changes(self) -> None:
        selected = self._changes_host().selected_run()
        context = session_view(selected).context
        identity = (selected.run_id, context) if selected and context else None
        changed = identity != self._changes_identity
        if changed:
            self._changes_identity = identity
            self._changes_token += 1
            self._diff_token += 1
            self._diff_identity = None
            self._pending_changes = self._pending_diff = None
            self._clear_changes(
                None if identity else "Changes unavailable: no session worktree.",
                loading=identity is not None,
            )
        if identity is None:
            return
        now = monotonic()
        if not changed and now - self._last_changes_request < 1.0:
            return
        self._last_changes_request = now
        request = _ChangesRequest(self._changes_token, *identity)
        if self._changes_busy:
            self._pending_changes = request
        else:
            self._start_changes(request)

    def _clear_changes(self, error: str | None, *, loading: bool) -> None:
        self._changes_files = ()
        self.selected_file_path = None
        self._changes_diff = ""
        self._changes_error = error
        self._changes_loading = loading
        self._render_changes()

    def _start_changes(self, request: _ChangesRequest) -> None:
        self._changes_busy = True
        _ = self._load_changes(request)

    @work(thread=True, group="git-changes", exclusive=False)
    def _load_changes(self, request: _ChangesRequest) -> None:
        try:
            files = GitAdapter(Path(request.context.cwd)).session_changes(request.context)
            error = None
        except (GitOperationError, OSError, ValueError) as exc:
            files = ()
            error = str(exc)
        _ = self._changes_host().call_from_thread(self._receive_changes, request, files, error)

    def _receive_changes(
        self, request: _ChangesRequest, files: tuple[ChangedFile, ...], error: str | None
    ) -> None:
        self._changes_busy = False
        pending, self._pending_changes = self._pending_changes, None
        if (
            request.token == self._changes_token
            and (request.run_id, request.context) == self._changes_identity
        ):
            self._changes_files = files
            self._changes_error = error
            self._changes_loading = False
            if self.selected_file_path not in {item.path for item in files}:
                self.selected_file_path = files[0].path if files else None
            self._render_changes()
            self._request_diff()
        if pending is not None:
            self._start_changes(pending)

    def _request_diff(self) -> None:
        identity = (
            (*self._changes_identity, self.selected_file_path)
            if self._changes_identity and self.selected_file_path
            else None
        )
        if identity != self._diff_identity:
            self._diff_identity = identity
            self._diff_token += 1
            self._pending_diff = None
            self._changes_diff = "Loading diff..." if identity else ""
            self._render_changes()
        if identity is None:
            return
        request = _DiffRequest(self._diff_token, *identity)
        if self._diff_busy:
            self._pending_diff = request
        else:
            self._start_diff(request)

    def _start_diff(self, request: _DiffRequest) -> None:
        self._diff_busy = True
        _ = self._load_diff(request)

    @work(thread=True, group="git-diff", exclusive=False)
    def _load_diff(self, request: _DiffRequest) -> None:
        try:
            diff = GitAdapter(Path(request.context.cwd)).session_diff(
                request.context, request.path
            )
            error = None
        except (GitOperationError, OSError, ValueError) as exc:
            diff = ""
            error = str(exc)
        _ = self._changes_host().call_from_thread(self._receive_diff, request, diff, error)

    def _receive_diff(self, request: _DiffRequest, diff: str, error: str | None) -> None:
        self._diff_busy = False
        pending, self._pending_diff = self._pending_diff, None
        if (
            request.token == self._diff_token
            and (request.run_id, request.context, request.path) == self._diff_identity
        ):
            self._changes_diff = diff
            self._changes_error = error
            self._render_changes()
        if pending is not None:
            self._start_diff(pending)

    def _render_changes(self) -> None:
        host = self._changes_host()
        if host.is_mounted:
            host.query_one("#changes-panel", ChangesPanel).update(
                ChangesPanelState(
                    files=self._changes_files,
                    selected_path=self.selected_file_path,
                    diff=self._changes_diff,
                    error=self._changes_error,
                    loading=self._changes_loading,
                )
            )

    @on(DataTable.RowSelected, "#changes-files")
    def select_changed_file(self, event: DataTable.RowSelected) -> None:
        path = str(event.row_key.value)
        if path in {item.path for item in self._changes_files}:
            self.selected_file_path = path
            self._request_diff()
