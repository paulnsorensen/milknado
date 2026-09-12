"""Session event index updates."""

from __future__ import annotations

from collections import deque
from contextlib import suppress

import msgspec

from milknado.domains.common import SessionEvent, SessionInput, normalize_session_event

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


def normalize_event(
    event: SessionEvent, event_index: dict[tuple[str, str], SessionEvent]
) -> SessionEvent:
    key = (event.kind, event.event_id)
    previous = event_index.get(key) if event.event_id else None
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


def normalize_input(command: SessionInput, prefix: str, token: int) -> SessionInput:
    command_id = (
        command.command_id
        or (
            command.request_id.removeprefix(prefix)
            if command.action in {"approve", "deny"}
            else command.request_id
        )
        or f"input-{token}"
    )
    if command.action in {"approve", "deny"}:
        return msgspec.structs.replace(
            command, request_id=command.request_id.removeprefix(prefix), command_id=command_id
        )
    if not command.request_id or command.command_id != command_id:
        return msgspec.structs.replace(command, request_id=command_id, command_id=command_id)
    return command


def update_indexes(  # noqa: PLR0913
    event: SessionEvent,
    event_index: dict[tuple[str, str], SessionEvent],
    permissions: dict[tuple[str, str], SessionEvent],
    pending: dict[int, SessionInput],
    inputs: deque[int],
    inflight: dict[str, SessionInput],
    prefix: str,
) -> None:
    key = (event.kind, event.event_id)
    if event.event_id:
        event_index[key] = event
    if event.kind == "permission":
        if event.state in {"requested", "submitted"}:
            permissions[key] = event
        elif event.state in {"approved", "denied", "cancelled"}:
            _ = permissions.pop(key, None)
            if event.state in {"approved", "denied"} and event.event_id:
                _acknowledge(event.event_id, prefix, pending, inputs)
                _acknowledge_inflight(event.event_id, prefix, inflight)
    if (
        event.kind == "user"
        and event.event_id
        and event.state in {"delivered", "rejected", "unconfirmed"}
    ):
        _acknowledge(event.event_id, prefix, pending, inputs)
        _acknowledge_inflight(event.event_id, prefix, inflight)


def receipt_state(event: SessionEvent) -> str:
    return {"approved": "delivered", "denied": "delivered"}.get(event.state, event.state)


def terminal_command(
    event: SessionEvent, prefix: str, inflight: dict[str, SessionInput]
) -> SessionInput | None:
    if event.kind == "user":
        if event.state not in {"delivered", "rejected", "unconfirmed"}:
            return None
    elif event.kind == "permission":
        if event.state not in {"approved", "denied"}:
            return None
    else:
        return None
    return next(
        (
            command
            for command in inflight.values()
            if command.command_id == event.event_id
            or prefix + command.command_id == event.event_id
            or prefix + command.request_id == event.event_id
        ),
        None,
    )


def _acknowledge(
    request_id: str,
    prefix: str,
    pending: dict[int, SessionInput],
    inputs: deque[int],
) -> None:
    tokens = tuple(
        token
        for token, command in pending.items()
        if (
            command.command_id == request_id
            or prefix + command.command_id == request_id
            or prefix + command.request_id == request_id
        )
    )
    for token in tokens:
        with suppress(ValueError):
            _ = inputs.remove(token)
        _ = pending.pop(token, None)


def _acknowledge_inflight(request_id: str, prefix: str, inflight: dict[str, SessionInput]) -> None:
    for key, command in tuple(inflight.items()):
        if (
            command.command_id == request_id
            or prefix + command.command_id == request_id
            or prefix + command.request_id == request_id
        ):
            _ = inflight.pop(key, None)


def remember(
    event: SessionEvent,
    events: deque[SessionEvent],
    event_index: dict[tuple[str, str], SessionEvent],
    max_events: int,
) -> None:
    key = (event.kind, event.event_id)
    previous = event_index.get(key) if event.event_id else None
    if previous is None:
        events.append(event)
        if len(events) > max_events:
            evicted = events.popleft()
            evicted_key = (evicted.kind, evicted.event_id)
            if event_index.get(evicted_key) == evicted:
                _ = event_index.pop(evicted_key, None)
        return
    for index, current in enumerate(events):
        if current == previous:
            events[index] = event
            break
