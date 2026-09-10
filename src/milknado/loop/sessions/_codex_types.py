from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

import msgspec

RequestId: TypeAlias = int | str


class CodexFrame(msgspec.Struct, frozen=True, kw_only=True):
    method: str | None = None
    id: int | str | None = None
    params: object | None = None
    result: object | None = None
    error: object | None = None


@dataclass(slots=True)
class PendingRequest:
    method: str
    stage: str
    event_id: str = ""
    text: str = ""
    turn_id: str = ""


@dataclass(slots=True)
class ApprovalRequest:
    raw_id: RequestId
    method: str
    params: dict[str, object]
    action: str = ""


def encode_line(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


CodexDeltaKind: TypeAlias = Literal["assistant", "tool"]


if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Protocol

    from milknado.loop.sessions._codex_policy import CodexPolicy
    from milknado.loop.sessions._protocol import ProtocolStep

    class CodexState(Protocol):
        _active: bool
        _approvals: dict[str, ApprovalRequest]
        _assistant: dict[str, str]
        _fatal: bool
        _initialized: bool
        _pending: dict[RequestId, PendingRequest]
        _policy: CodexPolicy
        _prompt: str
        _session_id: str
        _thread_id: str
        _turn_id: str

        def _approval_request(
            self, frame: CodexFrame, params: dict[str, object]
        ) -> ProtocolStep: ...

        def _approval_resolved(self, params: dict[str, object]) -> ProtocolStep: ...

        def _delta_event(
            self, params: dict[str, object], kind: CodexDeltaKind
        ) -> ProtocolStep: ...

        def _failure(self, message: str) -> ProtocolStep: ...

        def _item_event(self, params: dict[str, object], complete: bool) -> ProtocolStep: ...

        def _progress_event(self, params: dict[str, object]) -> ProtocolStep: ...

        def _request(
            self, pending: PendingRequest, params: Mapping[str, object]
        ) -> tuple[RequestId, bytes]: ...

        def _result(self, turn: dict[str, object]) -> str | None: ...

        def _start_turn(self, prompt: str, event_id: str = "") -> ProtocolStep: ...

        def _terminal_event(self, params: dict[str, object]) -> ProtocolStep: ...

        def _turn_completed(self, params: dict[str, object]) -> ProtocolStep: ...

        def _turn_error(self, params: dict[str, object]) -> ProtocolStep: ...
else:

    class CodexState:
        pass
