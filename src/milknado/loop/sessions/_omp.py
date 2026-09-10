"""Oh My Pi JSON-RPC session protocol."""

from __future__ import annotations

import msgspec

from milknado.domains.common import SessionEvent
from milknado.loop.sessions._omp_control import OmpControlMixin
from milknado.loop.sessions._omp_state import Pending
from milknado.loop.sessions._omp_wire import OmpFrame, encode, text
from milknado.loop.sessions._protocol import ProtocolStep


class OmpSession(OmpControlMixin):
    """Drive the documented OMP RPC protocol without spawning a process."""

    def start(self, prompt: str) -> ProtocolStep:
        if self._started:
            raise ValueError("OMP session already started")
        self._started: bool = True
        self._active: bool = True
        request_id = self._next_id("prompt")
        self._pending[request_id] = Pending("prompt", prompt)
        command = encode({"id": request_id, "type": "prompt", "message": prompt})
        events = (
            self._status("running", "OMP RPC session started", "lifecycle"),
            SessionEvent(kind="user", text=prompt, event_id=request_id, state="submitted"),
        )
        return ProtocolStep(commands=(command,), events=events)

    def receive(self, line: bytes) -> ProtocolStep:
        try:
            frame = msgspec.json.decode(line, type=OmpFrame)
        except (msgspec.DecodeError, msgspec.ValidationError, TypeError) as exc:
            return ProtocolStep(
                events=(self._error(f"Invalid OMP RPC frame: {exc}", "invalid_frame"),)
            )
        frame_type = frame.type
        if frame_type == "rpc_chunk":
            raw, error = self._chunks.consume(frame)
            if error:
                return ProtocolStep(events=(self._error(error, "invalid_chunk"),))
            return ProtocolStep() if raw is None else self.receive(raw)
        if frame_type == "ready":
            return self._ready(frame)
        if frame_type == "response":
            return self._response(frame)
        if frame_type == "extension_ui_request":
            return self._ui_request(frame)
        if frame_type in {"agent_start", "agent_end", "agent_settled"}:
            return self._agent_event(frame_type, frame)
        if frame_type in {"message_start", "message_update", "message_end"}:
            return self._message(frame_type, frame)
        if frame_type in {
            "tool_execution_start",
            "tool_execution_update",
            "tool_execution_end",
            "bash_execution_update",
        }:
            return self._tool(frame_type, frame)
        if frame_type == "prompt_result":
            return self._prompt_result(frame)
        if frame_type in {
            "queue_update",
            "compaction_start",
            "compaction_end",
            "auto_retry_start",
            "auto_retry_end",
            "advisor_cost_changed",
            "available_commands_update",
            "turn_start",
            "turn_end",
        }:
            return self._status_frame(frame_type, frame)
        if frame_type in {"error", "extension_error"}:
            self._failed: bool = True
            return self._reject(
                frame,
                text(frame.error) or text(frame.message) or f"OMP RPC {frame_type}",
                frame_type,
            )
        return ProtocolStep(
            events=(self._error(f"Unsupported OMP RPC event: {frame_type}", "unsupported_event"),)
        )

    def _ready(self, frame: OmpFrame) -> ProtocolStep:
        self._chunks.set_limit(frame.max_reassembled_frame_bytes)
        versions = frame.supported_protocol_versions
        commands: list[bytes] = []
        if isinstance(versions, list) and 2 in versions:
            protocol_id = "protocol-1"
            self._pending[protocol_id] = Pending("negotiate_protocol", "")
            commands.append(
                encode({"id": protocol_id, "type": "negotiate_protocol", "protocolVersion": 2})
            )
        state_id = "state-1"
        self._pending[state_id] = Pending("get_state", "")
        commands.append(encode({"id": state_id, "type": "get_state"}))
        return ProtocolStep(
            commands=tuple(commands),
            events=(self._status("ready", "OMP RPC transport ready", "lifecycle"),),
        )

    def _agent_event(self, frame_type: str, frame: OmpFrame) -> ProtocolStep:
        if frame_type == "agent_end":
            self._record_messages(frame.messages)
            waiting_for_queued = any(
                pending.action in {"steer", "follow_up"} for pending in self._pending.values()
            )
            interrupted = self._interrupt_accepted and self._saw_aborted
            terminal = frame.is_terminal is not False and (
                interrupted or (not waiting_for_queued and not self._queued)
            )
            event = self._status(
                "complete" if terminal else "running", "OMP agent turn complete", "lifecycle"
            )
            if not terminal:
                return ProtocolStep(events=(event,))
            self._active = False
            return ProtocolStep(
                events=(event,),
                done=True,
                result_text=self._result_text,
                failed=self._failed and (not interrupted),
                interrupted=interrupted,
                session_id=self._session_id,
            )
        if frame_type == "agent_settled":
            return ProtocolStep(
                events=(self._status("settled", "OMP session settled", "lifecycle"),)
            )
        return ProtocolStep(events=(self._status("running", "OMP agent started", "lifecycle"),))

    def _prompt_result(self, frame: OmpFrame) -> ProtocolStep:
        request_id = frame.id
        if frame.agent_invoked is False:
            self._active = False
            return ProtocolStep(
                events=(
                    self._status(
                        "complete", "OMP prompt completed without agent invocation", "lifecycle"
                    ),
                ),
                done=True,
                result_text=self._result_text,
                failed=self._failed,
                session_id=self._session_id,
            )
        return ProtocolStep(
            events=(
                self._status(
                    "queued",
                    "OMP prompt scheduled",
                    request_id if isinstance(request_id, str) else self._next_id("prompt"),
                ),
            )
        )
