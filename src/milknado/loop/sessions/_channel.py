from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from threading import RLock
from typing import final

import msgspec

from milknado.domains.common import (
    SessionAction,
    SessionContext,
    SessionEvent,
    SessionInput,
    SessionView,
)
from milknado.loop.sessions._capabilities import CapabilitySink, refresh
from milknado.loop.sessions._channel_indexes import (
    normalize_event,
    normalize_input,
    receipt_state,
    remember,
    terminal_command,
    update_indexes,
)

SessionSink = Callable[[SessionEvent], None]
_STREAM_FLUSH_INTERVAL = 0.05


@final
class SessionChannel:
    def __init__(  # noqa: PLR0913
        self,
        sink: SessionSink | None = None,
        *,
        command_state_sink: Callable[[SessionInput, str], None] | None = None,
        durable_drain: Callable[[], tuple[SessionInput, ...]] | None = None,
        capability_sink: CapabilitySink | None = None,
        max_events: int = 500,
        max_inputs: int = 64,
    ) -> None:
        if max_events < 1 or max_inputs < 1:
            raise ValueError("SessionChannel capacities must be positive")
        self._lock: RLock = RLock()
        self._sink: SessionSink | None = sink
        self._command_state_sink: Callable[[SessionInput, str], None] | None = command_state_sink
        self._durable_drain: Callable[[], tuple[SessionInput, ...]] | None = durable_drain
        self._capability_sink: CapabilitySink | None = capability_sink
        self._close_sink: Callable[[str], None] | None = None
        self._max_events, self._max_inputs = max_events, max_inputs
        self._events: deque[SessionEvent] = deque()
        self._event_index: dict[tuple[str, str], SessionEvent] = {}
        self._permissions: dict[tuple[str, str], SessionEvent] = {}
        self._inputs: deque[int] = deque()
        self._pending: dict[int, SessionInput] = {}
        self._inflight: dict[str, SessionInput] = {}
        self._next_token, self._epoch = 0, 0
        self._last_sink_at, self._prefix = 0.0, ""
        self._deferred: dict[tuple[str, str], SessionEvent] = {}
        self._context: SessionContext | None = None
        self._actions: tuple[SessionAction, ...] = ()
        self._invocation_id = ""
        self._active, self._closed, self._stopped_published = False, False, False

    def set_sink(self, sink: SessionSink | None) -> None:
        with self._lock:
            self._sink = sink

    def set_capability_sink(
        self, sink: CapabilitySink | None, *, on_close: Callable[[str], None] | None = None
    ) -> None:
        with self._lock:
            self._capability_sink = sink
            self._close_sink = on_close

    def start(
        self,
        context: SessionContext,
        actions: tuple[SessionAction, ...],
        *,
        invocation_id: str = "",
    ) -> None:
        with self._lock:
            if self._closed and (self._pending or self._deferred or self._permissions):
                raise RuntimeError("cannot restart a session with unpersisted terminal events")
            if self._active and self._context != context:
                raise RuntimeError("cannot replace an active session context")
            if not self._active:
                self._epoch += 1
                self._prefix = f"{self._epoch}/"
            self._closed = False
            self._context = context
            self._actions = tuple(dict.fromkeys(actions))
            self._invocation_id = invocation_id or self._invocation_id
            self._active = True
            self._stopped_published = False
        refresh(self._capability_sink, self._capability_state(context, invocation_id))

    def _capability_state(
        self, context: SessionContext | None, invocation_id: str = ""
    ) -> tuple[SessionContext | None, tuple[SessionAction, ...], str, tuple[str, ...]]:
        return (
            context,
            self._actions,
            invocation_id or self._invocation_id,
            tuple(event.event_id for event in self._permissions.values()),
        )

    def view(self) -> SessionView:
        with self._lock:
            return SessionView(
                context=self._context,
                events=tuple(self._events),
                actions=self._actions,
                active=self._active,
                permissions=tuple(self._permissions.values()),
            )

    def submit(self, command: SessionInput) -> bool:
        with self._lock:
            if not self._active or self._closed:
                self._record_user(command, "rejected")
                return False
            permission = self._permissions.get(("permission", command.request_id))
            if command.action not in self._actions or (
                command.action in {"approve", "deny"}
                and (permission is None or permission.state != "requested")
            ):
                self._record_user(command, "rejected")
                return False
            if len(self._pending) >= self._max_inputs:
                self._record_user(command, "rejected")
                return False
            token = self._next_token
            self._next_token += 1
            command = normalize_input(command, self._prefix, token)
            self._record_user(command, "queued")
            self._pending[token] = command
            self._inputs.append(token)
            return True

    def drain(self) -> tuple[SessionInput, ...]:
        submitted: list[SessionInput] = []
        if self._durable_drain is not None:
            durable = self._durable_drain()
            with self._lock:
                for command in durable:
                    if command.action in {"approve", "deny"}:
                        command = msgspec.structs.replace(
                            command, request_id=command.request_id.removeprefix(self._prefix)
                        )
                    key = command.command_id or command.request_id
                    if key:
                        self._inflight[key] = command
                    submitted.append(command)
        with self._lock:
            while self._inputs:
                token = self._inputs[0]
                command = self._pending[token]
                self._record_user(command, "submitted")
                _ = self._inputs.popleft()
                submitted.append(command)
        return tuple(submitted)

    def confirm(self, command: SessionInput, state: str) -> None:
        with self._lock:
            self._record_user(command, state)

    def publish(self, event: SessionEvent) -> None:
        with self._lock:
            if event.event_id:
                event = msgspec.structs.replace(event, event_id=self._prefix + event.event_id)
            normalized = normalize_event(event, self._event_index)
            key = (normalized.kind, normalized.event_id)
            previous = self._event_index.get(key) if normalized.event_id else None
            if normalized == previous:
                return
            if self._should_defer(event):
                self._deferred[key] = normalized
            else:
                self._flush_deferred()
                self._persist(normalized)
            remember(normalized, self._events, self._event_index, self._max_events)
            command = terminal_command(normalized, self._prefix, self._inflight)
            if command is not None and self._command_state_sink is not None:
                state = receipt_state(normalized)
                self._command_state_sink(command, state)
            update_indexes(
                normalized,
                self._event_index,
                self._permissions,
                self._pending,
                self._inputs,
                self._inflight,
                self._prefix,
            )
            if not self._closed or self._close_sink is None:
                refresh(self._capability_sink, self._capability_state(self._context))

    def close(self) -> None:
        first_error: Exception | None = None
        with self._lock:
            self._active = False
            self._closed = True
            try:
                self._flush_deferred()
            except Exception as exc:  # noqa: BLE001 - Finish cleanup before re-raising.
                first_error = exc
            queued = frozenset(self._inputs)
            for token, command in tuple(self._pending.items()):
                try:
                    state = "rejected" if token in queued else "unconfirmed"
                    self._record_user(command, state)
                except Exception as exc:  # noqa: BLE001 - Finish cleanup before re-raising.
                    first_error = first_error or exc
                    continue
                _ = self._pending.pop(token, None)
            for key, command in tuple(self._inflight.items()):
                try:
                    self._record_user(command, "unconfirmed")
                except Exception as exc:  # noqa: BLE001 - Finish cleanup before re-raising.
                    first_error = first_error or exc
                finally:
                    _ = self._inflight.pop(key, None)
            for permission in tuple(self._permissions.values()):
                try:
                    self.publish(
                        msgspec.structs.replace(
                            permission,
                            event_id=permission.event_id.removeprefix(self._prefix),
                            state="cancelled",
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - Finish cleanup before re-raising.
                    first_error = first_error or exc
            self._actions = ()
            if not self._stopped_published:
                stopped = SessionEvent(kind="status", text="session stopped", state="stopped")
                try:
                    if self._close_sink is not None and self._invocation_id:
                        self._close_sink(self._invocation_id)
                    self.publish(stopped)
                except Exception as exc:  # noqa: BLE001 - Finish cleanup before re-raising.
                    first_error = first_error or exc
                else:
                    self._stopped_published = True
        if first_error is not None:
            raise first_error

    def _record_user(self, command: SessionInput, state: str) -> None:
        durable = command in self._inflight.values()
        if self._command_state_sink is not None and not (
            durable and state in {"delivered", "rejected", "unconfirmed"}
        ):
            self._command_state_sink(command, state)
        self.publish(
            SessionEvent(
                kind="user",
                text=command.text,
                event_id=command.command_id or command.request_id,
                state=state,
                action=command.action,
            )
        )

    def _should_defer(self, event: SessionEvent) -> bool:
        return bool(
            self._sink is not None
            and event.event_id
            and event.delta
            and event.kind in {"assistant", "tool"}
            and event.state in {"", "streaming"}
            and time.monotonic() - self._last_sink_at < _STREAM_FLUSH_INTERVAL
        )

    def _flush_deferred(self) -> None:
        for key in tuple(self._deferred):
            event = self._deferred.pop(key, None)
            if event is None:
                continue
            try:
                self._persist(event)
            except Exception:
                self._deferred = {key: event} | self._deferred
                raise

    def _persist(self, event: SessionEvent) -> None:
        if self._sink is not None:
            self._sink(event)
            self._last_sink_at = time.monotonic()
