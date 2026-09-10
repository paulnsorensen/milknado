from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import msgspec

from milknado.domains.common.session import SessionEvent

from ._protocol import ProtocolStep


@dataclass(slots=True)
class Turn:
    request_id: str
    text: str


@dataclass(slots=True)
class Permission:
    text: str
    input: dict[str, object]


@dataclass(slots=True)
class Block:
    kind: str
    event_id: str
    name: str = ""


class ClaudeMessage(msgspec.Struct, frozen=True, kw_only=True):
    id: str | None = None
    role: str | None = None
    content: str | list[dict[str, object]] | None = None


class ClaudeContentBlock(msgspec.Struct, frozen=True, kw_only=True):
    type: str = ""
    id: str | None = None
    name: str | None = None
    text: str | None = None
    thinking: str | None = None
    input: dict[str, object] | None = None


class ClaudeDelta(msgspec.Struct, frozen=True, kw_only=True):
    type: str = ""
    text: str | None = None
    thinking: str | None = None


class ClaudeStreamEvent(msgspec.Struct, frozen=True, kw_only=True):
    type: str = ""
    index: int | None = None
    message: ClaudeMessage | None = None
    content_block: ClaudeContentBlock | None = None
    delta: ClaudeDelta | None = None


class ClaudeControlRequest(msgspec.Struct, frozen=True, kw_only=True):
    subtype: str = ""
    tool_name: str | None = None
    title: str | None = None
    display_name: str | None = None
    input: dict[str, object] | None = None


class ClaudeControlResponse(msgspec.Struct, frozen=True, kw_only=True):
    subtype: str = ""
    request_id: str | None = None
    response: dict[str, object] | None = None
    error: str | None = None


class ClaudeError(msgspec.Struct, frozen=True, kw_only=True):
    message: str | None = None


class ClaudeFrame(msgspec.Struct, frozen=True, kw_only=True):
    type: str
    subtype: str | None = None
    is_error: bool | None = None
    result: str | None = None
    errors: list[str] | None = None
    session_id: str | None = None
    uuid: str | None = None
    model: str | None = None
    message: ClaudeMessage | str | None = None
    request_id: str | None = None
    request: ClaudeControlRequest | None = None
    response: ClaudeControlResponse | None = None
    error: str | ClaudeError | None = None
    event: ClaudeStreamEvent | None = None
    stop_reason: str | None = None
    reason: str | None = None


def encode_line(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


class ClaudeState:
    _turns: deque[Turn]
    _pending_users: deque[Turn]
    _user_ids: set[str]
    _permissions: dict[str, Permission]
    _controls: dict[str, str]
    _assistant_text: dict[str, str]
    _blocks: dict[int, Block]
    _assistant_id: str | None
    _active: bool
    _interrupt_requested: bool
    _interrupt_acknowledged: bool

    def __init__(self) -> None:
        self._sequence: int = 0
        self._turns = deque()
        self._pending_users = deque()
        self._user_ids = set()
        self._permissions = {}
        self._controls = {}
        self._assistant_text = {}
        self._blocks = {}
        self._assistant_id = None
        self._active = False
        self._interrupt_requested = False
        self._interrupt_acknowledged = False

    def _id(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{self._sequence}"

    def _error(self, text: str, *, done: bool = True, failed: bool = True) -> ProtocolStep:
        if done:
            self._active = False
        event = SessionEvent(
            kind="error",
            text=text,
            event_id=self._id("error"),
            state="stopped" if done else "running",
        )
        return ProtocolStep(
            events=(event,), done=done, failed=failed, result_text=text if done else None
        )

    def _frame_id(self, raw: ClaudeFrame, prefix: str) -> str:
        return raw.uuid or raw.session_id or self._id(prefix)
