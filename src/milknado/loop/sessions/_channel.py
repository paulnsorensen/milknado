from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from threading import RLock

import msgspec

from milknado.domains.common import (
    SessionAction,
    SessionContext,
    SessionEvent,
    SessionInput,
    SessionView,
    normalize_session_event,
)

SessionSink = Callable[[SessionEvent], None]
_STREAM_FLUSH_INTERVAL = 0.05
_RECEIPT_KINDS = frozenset({"user", "permission"})
_RECEIPT_STATE_RANK = {
    "queued": 0,
    "requested": 0,
    "submitted": 1,
    "delivered": 2,
    "rejected": 2,
    "approved": 2,
    "denied": 2,
    "cancelled": 2,
    "unconfirmed": 2,
}


class SessionChannel:
    """Thread-safe bounded state and input admission for one worker session."""

    def __init__(
        self,
        sink: SessionSink | None = None,
        *,
        max_events: int = 500,
        max_inputs: int = 64,
    ) -> None:
        if max_events < 1 or max_inputs < 1:
            raise ValueError("SessionChannel capacities must be positive")
        self._lock: RLock = RLock()
        self._sink: SessionSink | None = sink
        self._max_events: int = max_events
        self._max_inputs: int = max_inputs
        self._events: deque[SessionEvent] = deque()
        self._event_index: dict[tuple[str, str], SessionEvent] = {}
        self._permissions: dict[tuple[str, str], SessionEvent] = {}
        self._inputs: deque[int] = deque()
        self._pending: dict[int, SessionInput] = {}
        self._next_token: int = 0
        self._epoch: int = 0
        self._prefix: str = ""
        self._deferred: dict[tuple[str, str], SessionEvent] = {}
        self._last_sink_at: float = 0.0
        self._context: SessionContext | None = None
        self._actions: tuple[SessionAction, ...] = ()
        self._active: bool = False
        self._closed: bool = False
        self._stopped_published: bool = False

    def set_sink(self, sink: SessionSink | None) -> None:
        """Replace the durable sink used for subsequent state changes."""
        with self._lock:
            self._sink = sink

    def start(self, context: SessionContext, actions: tuple[SessionAction, ...]) -> None:
        """Open or reopen the channel with the protocol's truthful actions."""
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
            self._active = True
            self._stopped_published = False

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
        """Admit a command only after its queued claim is durably recorded."""
        with self._lock:
            if not self._active or self._closed:
                self._record_user(command, "rejected")
                return False
            if command.action not in self._actions or not self._permission_is_pending(command):
                self._record_user(command, "rejected")
                return False
            if len(self._pending) >= self._max_inputs:
                self._record_user(command, "rejected")
                return False
            token = self._next_token
            self._next_token += 1
            if command.action in {"approve", "deny"}:
                command = msgspec.structs.replace(
                    command, request_id=command.request_id.removeprefix(self._prefix)
                )
            elif not command.request_id:
                command = msgspec.structs.replace(command, request_id=f"input-{token}")
            self._record_user(command, "queued")
            self._pending[token] = command
            self._inputs.append(token)
            return True

    def drain(self) -> tuple[SessionInput, ...]:
        """Move admitted commands to submitted state and return them once."""
        submitted: list[SessionInput] = []
        with self._lock:
            while self._inputs:
                token = self._inputs[0]
                command = self._pending[token]
                self._record_user(command, "submitted")
                _ = self._inputs.popleft()
                submitted.append(command)
        return tuple(submitted)

    def publish(self, event: SessionEvent) -> None:
        """Persist and coalesce one protocol event before exposing it in memory."""
        with self._lock:
            if event.event_id:
                event = msgspec.structs.replace(event, event_id=self._prefix + event.event_id)
            normalized = self._normalize(event)
            key = (normalized.kind, normalized.event_id)
            previous = self._event_index.get(key) if normalized.event_id else None
            if normalized == previous:
                return
            if self._should_defer(event):
                self._deferred[key] = normalized
            else:
                self._flush_deferred()
                self._persist(normalized)
            self._remember(normalized)
            self._update_indexes(normalized)

    def close(self) -> None:
        """Reject unsent inputs and mark inputs without receipts as unconfirmed."""
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
            if not self._stopped_published:
                stopped = SessionEvent(kind="status", text="session stopped", state="stopped")
                try:
                    self.publish(stopped)
                except Exception as exc:  # noqa: BLE001 - Finish cleanup before re-raising.
                    first_error = first_error or exc
                else:
                    self._stopped_published = True
        if first_error is not None:
            raise first_error

    def _permission_is_pending(self, command: SessionInput) -> bool:
        if command.action not in {"approve", "deny"}:
            return True
        if not command.request_id:
            return False
        permission = self._permissions.get(("permission", command.request_id))
        return permission is not None and permission.state == "requested"

    def _record_user(self, command: SessionInput, state: str) -> None:
        self.publish(
            SessionEvent(
                kind="user",
                text=command.text,
                event_id=command.request_id,
                state=state,
                action=command.action,
            )
        )

    def _normalize(self, event: SessionEvent) -> SessionEvent:
        key = (event.kind, event.event_id)
        previous = self._event_index.get(key) if event.event_id else None
        normalized = normalize_session_event(event, previous)
        if previous is None or event.kind not in _RECEIPT_KINDS:
            return normalized
        previous_rank = _RECEIPT_STATE_RANK.get(previous.state)
        current_rank = _RECEIPT_STATE_RANK.get(normalized.state)
        if previous_rank is None or current_rank is None:
            return normalized
        if current_rank < previous_rank or (
            previous_rank == current_rank == 2 and normalized.state != previous.state
        ):
            return previous
        return normalized

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

    def _remember(self, event: SessionEvent) -> None:
        key = (event.kind, event.event_id)
        previous = self._event_index.get(key) if event.event_id else None
        if previous is None:
            self._events.append(event)
            if len(self._events) > self._max_events:
                evicted = self._events.popleft()
                evicted_key = (evicted.kind, evicted.event_id)
                if self._event_index.get(evicted_key) == evicted:
                    _ = self._event_index.pop(evicted_key, None)
            return
        for index, current in enumerate(self._events):
            if current == previous:
                self._events[index] = event
                break

    def _update_indexes(self, event: SessionEvent) -> None:
        key = (event.kind, event.event_id)
        if event.event_id:
            self._event_index[key] = event
        if event.kind == "permission":
            if event.state in {"requested", "submitted"}:
                self._permissions[key] = event
            elif event.state in {"approved", "denied", "cancelled"}:
                _ = self._permissions.pop(key, None)
                if event.state in {"approved", "denied"} and event.event_id:
                    self._acknowledge(event.event_id)
        if event.kind == "user" and event.event_id and event.state in {"delivered", "rejected"}:
            self._acknowledge(event.event_id)

    def _acknowledge(self, request_id: str) -> None:
        tokens = tuple(
            token
            for token, command in self._pending.items()
            if self._prefix + command.request_id == request_id
        )
        for token in tokens:
            with suppress(ValueError):
                _ = self._inputs.remove(token)
            _ = self._pending.pop(token, None)
