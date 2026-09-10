from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import msgspec

SessionAction = Literal["steer", "follow_up", "interrupt", "approve", "deny"]
SessionKind = Literal["assistant", "tool", "user", "error", "status", "permission"]


class SessionInput(msgspec.Struct, frozen=True, kw_only=True):
    action: SessionAction
    text: str = ""
    request_id: str = ""


class SessionEvent(msgspec.Struct, frozen=True, kw_only=True):
    kind: SessionKind
    text: str
    event_id: str = ""
    state: str = ""
    delta: bool = False
    action: SessionAction | None = None


class SessionContext(msgspec.Struct, frozen=True, kw_only=True):
    family: str
    cwd: str
    base_oid: str = ""


@dataclass(frozen=True, slots=True)
class SessionView:
    context: SessionContext | None = None
    events: tuple[SessionEvent, ...] = ()
    actions: tuple[SessionAction, ...] = ()
    active: bool = False
    permissions: tuple[SessionEvent, ...] = ()


def normalize_session_event(
    event: SessionEvent, previous: SessionEvent | None = None
) -> SessionEvent:
    """Bound display output and preserve the identity of acknowledged human input."""
    text = event.text
    action = event.action
    state = event.state
    if previous is not None:
        if event.delta:
            text = previous.text + text
            state = state or previous.state
        elif event.kind == "user" and not text:
            text = previous.text
        action = action or previous.action
    if event.kind not in {"user", "permission"} and len(text) > 8192:
        marker = "[Earlier text omitted]\n"
        text = marker + text[-(8192 - len(marker)) :]
    return msgspec.structs.replace(event, text=text, action=action, state=state, delta=False)
