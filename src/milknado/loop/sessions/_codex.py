from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import msgspec
from typing_extensions import override

from milknado import __version__
from milknado.domains.common import SessionAction, SessionEvent, SessionInput
from milknado.loop.sessions._codex_approval import CodexApprovalMixin, approval_result
from milknado.loop.sessions._codex_events import CodexEventMixin
from milknado.loop.sessions._codex_policy import CodexPolicy, translate_argv
from milknado.loop.sessions._codex_stream import CodexStreamMixin
from milknado.loop.sessions._codex_types import (
    ApprovalRequest,
    CodexFrame,
    PendingRequest,
    RequestId,
    encode_line,
)
from milknado.loop.sessions._protocol import ProtocolStep


class CodexSession(CodexEventMixin, CodexApprovalMixin, CodexStreamMixin):
    """Drive Codex 0.153.x's app-server JSON-RPC protocol over stdio."""

    actions: tuple[SessionAction, ...] = ("steer", "interrupt", "approve", "deny")
    command: tuple[str, ...]
    _policy: CodexPolicy
    _next_id: int
    _pending: dict[RequestId, PendingRequest]
    _approvals: dict[str, ApprovalRequest]
    _thread_id: str
    _turn_id: str
    _session_id: str
    _prompt: str
    _active: bool
    _initialized: bool
    _started: bool
    _fatal: bool
    _assistant: dict[str, str]

    def __init__(self, argv: tuple[str, ...], cwd: Path) -> None:
        self._policy = translate_argv(argv, cwd)
        self.command = self._policy.command
        self._next_id = 1
        self._pending = {}
        self._approvals = {}
        self._thread_id = ""
        self._turn_id = ""
        self._session_id = ""
        self._prompt = ""
        self._active = False
        self._initialized = False
        self._started = False
        self._fatal = False
        self._assistant = {}

    @override
    def _request(
        self, pending: PendingRequest, params: Mapping[str, object]
    ) -> tuple[RequestId, bytes]:
        request_id = self._next_id
        self._next_id += 1
        self._pending[request_id] = pending
        payload: dict[str, object] = {
            "method": pending.method,
            "id": request_id,
            "params": dict(params),
        }
        return request_id, encode_line(payload)

    def start(self, prompt: str) -> ProtocolStep:
        if self._started:
            raise ValueError("CodexSession.start() may only be called once")
        self._started = True
        self._prompt = prompt
        _request_id, command = self._request(
            PendingRequest("initialize", "initialize"),
            {
                "clientInfo": {
                    "name": "milknado",
                    "title": "Milknado",
                    "version": __version__,
                }
            },
        )
        return ProtocolStep(commands=(command,))

    def receive(self, line: bytes) -> ProtocolStep:
        try:
            frame = msgspec.json.decode(line, type=CodexFrame)
        except msgspec.DecodeError as exc:
            return self._failure(f"invalid Codex protocol frame: {exc}")
        if frame.method is not None:
            return self._notification(frame)
        if frame.id is not None:
            return self._response(frame)
        return self._failure("Codex protocol frame has neither method nor id")

    @override
    def _start_turn(self, prompt: str, event_id: str = "") -> ProtocolStep:
        request_id = self._next_id
        event_id = event_id or str(request_id)
        inputs: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        inputs.extend({"type": "localImage", "path": path} for path in self._policy.images)
        params: dict[str, object] = {
            "threadId": self._thread_id,
            "input": inputs,
            "clientUserMessageId": event_id,
        }
        params.update(self._policy.turn)
        _request_id, command = self._request(
            PendingRequest("turn/start", "turn", event_id=event_id, text=prompt),
            params,
        )
        event = SessionEvent(kind="user", text=prompt, event_id=event_id, state="submitted")
        return ProtocolStep(
            commands=(command,),
            events=(event,),
            session_id=self._session_id or None,
        )

    def submit(self, command: SessionInput) -> ProtocolStep:
        if self._fatal or not self._initialized or not self._thread_id:
            raise ValueError("Codex session is not accepting input")
        if command.action == "steer":
            return self._submit_steer(command)
        if command.action == "interrupt":
            return self._submit_interrupt()
        if command.action in {"approve", "deny"}:
            return self._submit_approval(command)
        raise ValueError(f"Codex does not support session action {command.action!r}")

    def _submit_steer(self, command: SessionInput) -> ProtocolStep:
        if not self._active or not self._turn_id:
            raise ValueError("Codex turn is not active")
        event_id = command.request_id or f"input-{self._next_id}"
        params: dict[str, object] = {
            "threadId": self._thread_id,
            "input": [{"type": "text", "text": command.text}],
            "expectedTurnId": self._turn_id,
            "clientUserMessageId": event_id,
        }
        _request_id, payload = self._request(
            PendingRequest(
                "turn/steer",
                "steer",
                event_id=event_id,
                text=command.text,
                turn_id=self._turn_id,
            ),
            params,
        )
        return ProtocolStep(commands=(payload,), session_id=self._session_id or None)

    def _submit_interrupt(self) -> ProtocolStep:
        if not self._active or not self._turn_id:
            raise ValueError("Codex turn is not active")
        params = {"threadId": self._thread_id, "turnId": self._turn_id}
        _request_id, payload = self._request(
            PendingRequest("turn/interrupt", "interrupt", turn_id=self._turn_id),
            params,
        )
        return ProtocolStep(commands=(payload,), session_id=self._session_id or None)

    def _submit_approval(self, command: SessionInput) -> ProtocolStep:
        approval = self._approvals.get(command.request_id)
        if approval is None or approval.action:
            raise ValueError("unknown or already resolved Codex approval request")
        result = approval_result(approval, command)
        approval.action = command.action
        return ProtocolStep(
            commands=(encode_line({"id": approval.raw_id, "result": result}),),
            session_id=self._session_id or None,
        )


__all__ = ["CodexSession"]
