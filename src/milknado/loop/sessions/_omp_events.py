"""Readable event translation for OMP RPC frames."""

from __future__ import annotations

from milknado.domains.common import SessionEvent
from milknado.loop.sessions._omp_state import OmpState
from milknado.loop.sessions._omp_wire import OmpFrame, OmpMessage, text
from milknado.loop.sessions._protocol import ProtocolStep


class OmpEventMixin(OmpState):
    _assistant_id: str
    _assistant_text: str
    _failed: bool
    _result_text: str | None
    _saw_aborted: bool
    _sequence: int

    def _message_update(self, frame: OmpFrame) -> ProtocolStep:
        event = frame.assistant_message_event
        if event is None:
            return ProtocolStep(
                events=(
                    self._error("OMP message_update has no assistant event", "invalid_message"),
                )
            )
        event_type = event.type
        event_id = self._assistant_id or self._next_id("assistant")
        self._assistant_id = event_id
        if event_type in {"thinking_start", "thinking_delta", "thinking_end", "toolcall_delta"}:
            return ProtocolStep()
        if event_type == "text_delta":
            value = text(event.delta)
            self._assistant_text += value
            return ProtocolStep(
                events=(
                    SessionEvent(
                        kind="assistant",
                        text=value,
                        event_id=event_id,
                        state="streaming",
                        delta=True,
                    ),
                )
            )
        if event_type == "text_end":
            value = text(event.content)
            self._assistant_text = value or self._assistant_text
            return ProtocolStep(
                events=(
                    SessionEvent(
                        kind="assistant",
                        text=self._assistant_text,
                        event_id=event_id,
                        state="streaming",
                    ),
                )
            )
        if event_type == "error":
            self._failed = True
            return ProtocolStep(
                events=(
                    self._error(
                        text(event.error) or "OMP assistant stream failed",
                        "assistant_error",
                        event_id,
                    ),
                ),
                failed=True,
            )
        if event_type == "done":
            message = event.message
            if message is not None:
                self._result_text = text(message.content) or self._result_text
        return ProtocolStep(
            events=(
                self._status("streaming", f"OMP assistant {event_type or 'update'}", event_id),
            )
        )

    def _message(self, frame_type: str, frame: OmpFrame) -> ProtocolStep:
        if frame_type == "message_update":
            return self._message_update(frame)
        message = frame.message if isinstance(frame.message, OmpMessage) else None
        if message is None:
            return ProtocolStep(
                events=(self._error(f"OMP {frame_type} has no message", "invalid_message"),)
            )
        role = message.role
        if role == "assistant" and message.stop_reason == "aborted":
            self._saw_aborted = True
        if role == "user":
            return self._user_echo(message) if frame_type == "message_start" else ProtocolStep()
        if role == "custom":
            if frame_type != "message_start":
                return ProtocolStep()
            kind = "tool" if message.custom_type == "async-result" else "status"
            event = SessionEvent(
                kind=kind,
                text=text(message.content),
                event_id=self._next_id("custom"),
                state="complete",
            )
            return ProtocolStep(events=(event,))
        if role == "assistant":
            return self._assistant_message(frame_type, message)
        if role == "toolResult":
            event_id = (
                message.tool_call_id
                if isinstance(message.tool_call_id, str)
                else self._next_id("tool")
            )
            state = "complete" if frame_type == "message_end" else "streaming"
            return ProtocolStep(
                events=(
                    SessionEvent(
                        kind="tool", text=text(message.content), event_id=event_id, state=state
                    ),
                )
            )
        return ProtocolStep(
            events=(self._error(f"Unsupported OMP message role: {role}", "unsupported_message"),)
        )

    def _user_echo(self, message: OmpMessage) -> ProtocolStep:
        value = text(message.content)
        for request_id, pending in self._pending.items():
            if (
                not pending.acknowledged
                and pending.action in {"prompt", "steer", "follow_up"}
                and pending.text == value
            ):
                pending.acknowledged = True
                return ProtocolStep(
                    events=(
                        SessionEvent(
                            kind="user", text=value, event_id=request_id, state="delivered"
                        ),
                    )
                )
        for request_id, (_, queued_text) in tuple(self._queued.items()):
            if queued_text == value:
                del self._queued[request_id]
                return ProtocolStep(
                    events=(
                        SessionEvent(
                            kind="user", text=value, event_id=request_id, state="delivered"
                        ),
                    )
                )
        return ProtocolStep(
            events=(
                SessionEvent(
                    kind="user", text=value, event_id=self._next_id("user"), state="delivered"
                ),
            )
        )

    def _assistant_message(self, frame_type: str, message: OmpMessage) -> ProtocolStep:
        if frame_type == "message_start":
            self._assistant_id = (
                message.id if isinstance(message.id, str) else self._next_id("assistant")
            )
            self._assistant_text = text(message.content)
        else:
            self._assistant_text = text(message.content) or self._assistant_text
        state = "complete" if frame_type == "message_end" else "streaming"
        event = SessionEvent(
            kind="assistant", text=self._assistant_text, event_id=self._assistant_id, state=state
        )
        if frame_type == "message_end":
            self._result_text = self._assistant_text or self._result_text
            stop_reason = message.stop_reason
            if stop_reason == "aborted":
                self._saw_aborted = True
            if stop_reason in {"error", "aborted"}:
                self._failed = True
                return ProtocolStep(
                    events=(
                        event,
                        self._error(
                            text(message.error_message) or str(stop_reason),
                            "assistant_error",
                            self._assistant_id,
                        ),
                    )
                )
        return ProtocolStep(events=(event,))

    def _tool(self, frame_type: str, frame: OmpFrame) -> ProtocolStep:
        event_id = frame.tool_call_id or frame.id or self._next_id("tool")
        if frame_type == "tool_execution_start":
            return ProtocolStep(
                events=(
                    SessionEvent(
                        kind="tool",
                        text=text(frame.tool_name) or "tool",
                        event_id=event_id,
                        state="streaming",
                    ),
                )
            )
        if frame_type == "tool_execution_end":
            state = "error" if frame.is_error is True else "complete"
            return ProtocolStep(
                events=(
                    SessionEvent(
                        kind="tool", text=text(frame.result), event_id=event_id, state=state
                    ),
                )
            )
        return ProtocolStep(
            events=(
                SessionEvent(
                    kind="tool",
                    text=text(frame.partial_result) or text(frame.delta),
                    event_id=event_id,
                    state="streaming",
                    delta=True,
                ),
            )
        )

    def _status_frame(self, frame_type: str, frame: OmpFrame) -> ProtocolStep:
        value = (
            text(frame.error_message) or text(frame.status_text) or frame_type.replace("_", " ")
        )
        failed = frame_type == "auto_retry_end" and frame.success is False
        return ProtocolStep(
            events=(
                self._status("error" if failed else "running", value, self._next_id("status")),
            ),
            failed=failed,
        )

    def _record_messages(self, messages: list[OmpMessage] | None) -> None:
        if messages is None:
            return
        for message in messages:
            if message.role != "assistant":
                continue
            value = text(message.content)
            if value:
                self._result_text = value
            if message.stop_reason == "aborted":
                self._saw_aborted = True

    def _status(self, state: str, value: str, event_id: str) -> SessionEvent:
        return SessionEvent(kind="status", text=value, event_id=event_id, state=state)

    def _error(self, value: str, state: str, event_id: str | None = None) -> SessionEvent:
        return SessionEvent(
            kind="error", text=value, event_id=event_id or self._next_id("error"), state=state
        )

    def _next_id(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{self._sequence}"
