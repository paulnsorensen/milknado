from __future__ import annotations

from milknado.domains.common.session import SessionEvent
from milknado.loop.sessions._protocol import ProtocolStep

from ._claude_state import (
    ClaudeControlRequest,
    ClaudeError,
    ClaudeFrame,
    ClaudeState,
    Permission,
    encode_line,
)


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


class ClaudeControlMixin(ClaudeState):
    _active: bool
    _interrupt_requested: bool
    _interrupt_acknowledged: bool

    def _result(self, raw: ClaudeFrame) -> ProtocolStep:
        result, result_id = self._result_text(raw), self._frame_id(raw, "result")
        interrupted = self._stop_marker(raw, result)
        failed = (
            bool(raw.is_error is True or raw.subtype not in (None, "success")) and not interrupted
        )
        if self._turns:
            _ = self._turns.popleft()
        self._interrupt_requested = self._interrupt_acknowledged = False
        if self._turns:
            return self._queued_result(result, result_id, failed, raw.session_id)
        return self._terminal_result(raw, result, result_id, interrupted)

    def _queued_result(
        self, result: str, result_id: str, failed: bool, session_id: str | None
    ) -> ProtocolStep:
        if failed:
            event = SessionEvent(kind="error", text=result, event_id=result_id, state="running")
        elif result:
            self._assistant_text[result_id] = result
            event = SessionEvent(
                kind="assistant", text=result, event_id=result_id, state="complete"
            )
        else:
            event = SessionEvent(
                kind="status",
                text="Turn complete; continuing queued input",
                event_id=result_id,
                state="running",
            )
        return ProtocolStep(events=(event,), session_id=session_id)

    def _terminal_result(
        self,
        raw: ClaudeFrame,
        result: str,
        result_id: str,
        interrupted: bool,
    ) -> ProtocolStep:
        self._active = False
        failed = (
            bool(raw.is_error is True or raw.subtype not in (None, "success")) and not interrupted
        )
        if failed:
            events = (
                SessionEvent(kind="error", text=result, event_id=result_id, state="stopped"),
            )
        elif result and result not in self._assistant_text.values():
            events = (
                SessionEvent(kind="assistant", text=result, event_id=result_id, state="complete"),
            )
        else:
            events = ()
        return ProtocolStep(
            events=events,
            done=True,
            result_text=result,
            failed=failed,
            interrupted=interrupted,
            session_id=raw.session_id,
        )

    def _control_request(self, raw: ClaudeFrame) -> ProtocolStep:
        request_id, request = raw.request_id, raw.request
        if not request_id or request is None:
            return self._error("Claude control request is missing request_id")
        if request.subtype != "can_use_tool":
            error = f"Unsupported Claude control request: {request.subtype or 'unknown'}"
            wire = {
                "type": "control_response",
                "response": {"subtype": "error", "request_id": request_id, "error": error},
            }
            event = SessionEvent(kind="error", text=error, event_id=request_id, state="running")
            return ProtocolStep(commands=(encode_line(wire),), events=(event,))
        text = self._permission_text(request)
        if request_id in self._permissions:
            return self._error(
                f"Duplicate Claude permission request_id: {request_id}", done=False, failed=False
            )
        self._permissions[request_id] = Permission(text, request.input or {})
        event = SessionEvent(kind="permission", text=text, event_id=request_id, state="requested")
        return ProtocolStep(events=(event,))

    def _control_response(self, raw: ClaudeFrame) -> ProtocolStep:
        response = raw.response
        request_id = response.request_id if response else None
        action = self._controls.pop(request_id, None) if request_id else None
        if not request_id or action is None:
            return self._error("Unexpected Claude control response", done=False, failed=False)
        if response and response.subtype != "success":
            if action == "interrupt":
                self._interrupt_requested = False
            event = SessionEvent(
                kind="error",
                text=response.error or "Claude control request failed",
                event_id=request_id,
                state="running",
            )
            return ProtocolStep(events=(event,))
        if action == "interrupt":
            self._interrupt_acknowledged = True
        text = "Claude session initialized" if action == "initialize" else "Interrupt acknowledged"
        return ProtocolStep(
            events=(SessionEvent(kind="status", text=text, event_id=request_id, state="running"),)
        )

    def _control_cancel(self, raw: ClaudeFrame) -> ProtocolStep:
        request_id = raw.request_id
        if request_id is None:
            return ProtocolStep()
        permission = self._permissions.pop(request_id, None)
        if permission is None:
            return ProtocolStep()
        event = SessionEvent(
            kind="permission",
            text=f"Permission cancelled: {permission.text}",
            event_id=request_id,
            state="denied",
        )
        return ProtocolStep(events=(event,))

    def _system(self, raw: ClaudeFrame) -> ProtocolStep:
        subtype = raw.subtype or "system"
        if subtype in ("error", "failure"):
            return self._error(self._error_text(raw))
        text = (
            f"Claude session started ({raw.model})"
            if subtype == "init" and raw.model
            else f"Claude {subtype}"
        )
        event = SessionEvent(
            kind="status",
            text=text,
            event_id=self._frame_id(raw, subtype),
            state="running",
        )
        return ProtocolStep(events=(event,), session_id=raw.session_id)

    def _error_text(self, raw: ClaudeFrame) -> str:
        if isinstance(raw.error, str) and raw.error:
            return raw.error
        if isinstance(raw.error, ClaudeError) and raw.error.message:
            return raw.error.message
        if raw.errors:
            return "; ".join(raw.errors)
        if isinstance(raw.message, str) and raw.message:
            return raw.message
        return "Claude returned an error"

    def _result_text(self, raw: ClaudeFrame) -> str:
        if raw.result is not None:
            return raw.result
        if raw.errors:
            return "; ".join(raw.errors)
        return raw.subtype or "Claude returned an empty result"

    def _stop_marker(self, raw: ClaudeFrame, result: str) -> bool:
        values = (raw.subtype, raw.stop_reason, raw.reason)
        markers = {"interrupted", "aborted", "cancelled", "canceled"}
        explicit = any(isinstance(value, str) and value.lower() in markers for value in values)
        text_marker = result.strip().lower().startswith(tuple(markers))
        return explicit or (self._interrupt_acknowledged and text_marker)

    def _permission_text(self, request: ClaudeControlRequest) -> str:
        title = request.title or request.display_name
        tool, data = request.tool_name or "tool", request.input or {}
        detail = _string(data.get("command")) or _string(data.get("file_path"))
        return title or (
            f"{tool}: {detail}" if detail else f"Claude requests permission to use {tool}"
        )
