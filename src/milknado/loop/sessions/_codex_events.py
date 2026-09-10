from __future__ import annotations

import json
from abc import ABCMeta
from typing import cast

from typing_extensions import override

from milknado.domains.common import SessionEvent
from milknado.loop.sessions._codex_types import (
    CodexFrame,
    CodexState,
    PendingRequest,
    encode_line,
)
from milknado.loop.sessions._protocol import ProtocolStep


def object_value(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def text_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def error_text(error: object) -> str:
    data = object_value(error)
    message = data.get("message") or data.get("error") or data.get("reason")
    return text_value(message) or text_value(error) or "Codex protocol error"


class CodexEventMixin(CodexState, metaclass=ABCMeta):
    _fatal: bool
    _active: bool
    _initialized: bool
    _thread_id: str
    _session_id: str
    _turn_id: str

    @override
    def _failure(self, message: str) -> ProtocolStep:
        self._fatal = True
        self._active = False
        event = SessionEvent(kind="error", text=message, event_id="codex", state="rejected")
        return ProtocolStep(
            events=(event,), done=True, failed=True, session_id=self._session_id or None
        )

    def _response(self, frame: CodexFrame) -> ProtocolStep:
        raw_id = frame.id
        if not isinstance(raw_id, (int, str)) or isinstance(raw_id, bool):
            return self._failure("Codex response id must be a string or integer")
        pending = self._pending.pop(raw_id, None)
        if pending is None:
            event = SessionEvent(
                kind="error",
                text=f"unmatched Codex response id {raw_id!r}",
                event_id=str(raw_id),
                state="rejected",
            )
            return ProtocolStep(events=(event,), session_id=self._session_id or None)
        if frame.error is not None:
            return self._request_error(pending, error_text(frame.error))
        result = object_value(frame.result)
        if not isinstance(frame.result, dict):
            return self._failure("Codex response result must be an object")
        return self._request_success(pending, result)

    def _request_error(self, pending: PendingRequest, message: str) -> ProtocolStep:
        events: list[SessionEvent] = []
        if pending.event_id:
            events.append(
                SessionEvent(
                    kind="user", text=pending.text, event_id=pending.event_id, state="rejected"
                )
            )
        events.append(
            SessionEvent(kind="error", text=message, event_id=pending.event_id, state="rejected")
        )
        fatal = pending.stage not in {"steer", "interrupt"}
        if fatal:
            self._fatal = True
            self._active = False
        return ProtocolStep(
            events=tuple(events), done=fatal, failed=fatal, session_id=self._session_id or None
        )

    def _request_success(self, pending: PendingRequest, result: dict[str, object]) -> ProtocolStep:
        if pending.stage == "initialize":
            self._initialized = True
            method = "thread/resume" if self._policy.resume_id else "thread/start"
            params: dict[str, object] = (
                {"threadId": self._policy.resume_id}
                if self._policy.resume_id
                else dict(self._policy.thread)
            )
            if self._policy.resume_id:
                params.update(self._policy.thread)
                params["threadId"] = self._policy.resume_id
            _request_id, command = self._request(PendingRequest(method, "thread"), params)
            initialized = encode_line({"method": "initialized", "params": {}})
            return ProtocolStep(commands=(initialized, command))
        if pending.stage == "thread":
            thread = object_value(result.get("thread"))
            self._thread_id = text_value(thread.get("id"))
            self._session_id = text_value(thread.get("sessionId")) or self._thread_id
            if not self._thread_id:
                return self._failure("Codex thread response omitted thread.id")
            step = self._start_turn(self._prompt)
            return ProtocolStep(
                commands=step.commands,
                events=step.events,
                session_id=self._session_id or None,
            )
        if pending.stage == "turn":
            turn = object_value(result.get("turn"))
            self._turn_id = text_value(turn.get("id")) or self._turn_id
            if not self._turn_id:
                return self._failure("Codex turn response omitted turn.id")
            status = text_value(turn.get("status"))
            self._active = status in {"", "inProgress", "running"}
            interrupted = status == "interrupted"
            events = (
                SessionEvent(
                    kind="user", text=pending.text, event_id=pending.event_id, state="delivered"
                ),
                SessionEvent(
                    kind="status",
                    text="turn running" if self._active else "turn complete",
                    event_id=self._turn_id,
                    state="running" if self._active else "complete",
                ),
            )
            return ProtocolStep(
                events=events,
                done=not self._active,
                result_text=self._result(turn),
                interrupted=interrupted,
                session_id=self._session_id or None,
            )
        if pending.stage == "steer":
            if not self._active or self._turn_id != pending.turn_id:
                event = SessionEvent(
                    kind="user", text=pending.text, event_id=pending.event_id, state="rejected"
                )
                return ProtocolStep(events=(event,), session_id=self._session_id or None)
            event = SessionEvent(
                kind="user", text=pending.text, event_id=pending.event_id, state="delivered"
            )
            return ProtocolStep(events=(event,), session_id=self._session_id or None)
        return ProtocolStep(session_id=self._session_id or None)

    def _notification(self, frame: CodexFrame) -> ProtocolStep:
        method = frame.method or ""
        params = object_value(frame.params)
        if method == "thread/started":
            thread = object_value(params.get("thread"))
            self._thread_id = text_value(thread.get("id")) or self._thread_id
            self._session_id = (
                text_value(thread.get("sessionId")) or self._session_id or self._thread_id
            )
            return ProtocolStep(session_id=self._session_id or None)
        if method == "serverRequest/resolved":
            return self._approval_resolved(params)
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
            "tool/requestUserInput",
            "mcpServer/elicitation/request",
            "execCommandApproval",
            "applyPatchApproval",
        }:
            return self._approval_request(frame, params)
        if method == "item/agentMessage/delta":
            return self._delta_event(params, "assistant")
        if method in {
            "item/commandExecution/outputDelta",
            "item/commandExecution/output_delta",
            "item/fileChange/outputDelta",
            "item/plan/delta",
            "item/reasoning/summaryTextDelta",
            "item/reasoning/textDelta",
        }:
            return self._delta_event(
                params, "assistant" if "plan" in method or "reasoning" in method else "tool"
            )
        if method == "item/mcpToolCall/progress":
            return self._progress_event(params)
        if method == "item/commandExecution/terminalInteraction":
            return self._terminal_event(params)
        if method in {"item/started", "item/completed"}:
            return self._item_event(params, method == "item/completed")
        if method == "turn/started":
            turn = object_value(params.get("turn"))
            self._turn_id = text_value(turn.get("id")) or self._turn_id
            self._active = True
            event = SessionEvent(
                kind="status", text="turn running", event_id=self._turn_id, state="running"
            )
            return ProtocolStep(events=(event,), session_id=self._session_id or None)
        if method == "turn/completed":
            return self._turn_completed(params)
        if method == "error":
            return self._turn_error(params)
        if frame.id is not None:
            return self._failure(f"Unsupported Codex server request: {method}")
        return ProtocolStep(session_id=self._session_id or None)
