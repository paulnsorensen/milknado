from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import msgspec

from milknado.domains.common.session import SessionAction, SessionEvent, SessionInput
from milknado.loop.sessions._protocol import ProtocolStep

from ._claude_control import ClaudeControlMixin
from ._claude_events import ClaudeEventsMixin
from ._claude_state import ClaudeFrame, ClaudeState, Turn, encode_line


def _with_value(args: list[str], flag: str, value: str) -> None:
    for index, argument in enumerate(args):
        if argument == flag:
            if index + 1 < len(args):
                args[index + 1] = value
            else:
                args.append(value)
            return
        if argument.startswith(f"{flag}="):
            args[index] = f"{flag}={value}"
            return
    args.extend((flag, value))


def _flag(args: list[str], flag: str) -> None:
    if flag not in args:
        args.append(flag)


class ClaudeSession(ClaudeEventsMixin, ClaudeControlMixin):
    """Claude Code's persistent stream-json input/output protocol."""

    actions: tuple[SessionAction, ...] = ("follow_up", "interrupt", "approve", "deny")
    _active: bool
    _interrupt_requested: bool

    def __init__(self, argv: tuple[str, ...], cwd: Path) -> None:
        if not argv:
            raise ValueError("ClaudeSession requires a Claude command")
        ClaudeState.__init__(self)
        self.cwd: Path = cwd
        command = list(argv)
        _with_value(command, "--output-format", "stream-json")
        _with_value(command, "--input-format", "stream-json")
        _flag(command, "--verbose")
        _flag(command, "--include-partial-messages")
        _flag(command, "--replay-user-messages")
        self.command: tuple[str, ...] = tuple(command)
        self._started: bool = False
        self._session_id: str | None = None
        self._buffer: bytearray = bytearray()

    def start(self, prompt: str) -> ProtocolStep:
        if self._started:
            raise ValueError("ClaudeSession has already started")
        self._started = self._active = True
        turn = Turn(self._id("user"), prompt)
        self._user_ids.add(turn.request_id)
        self._turns.append(turn)
        self._pending_users.append(turn)
        init_id = self._id("initialize")
        self._controls[init_id] = "initialize"
        return ProtocolStep(
            commands=(
                self._control(init_id, {"subtype": "initialize", "hooks": None}),
                self._user(prompt),
            ),
            events=(
                SessionEvent(
                    kind="user", text=prompt, event_id=turn.request_id, state="submitted"
                ),
            ),
        )

    def _user(self, text: str) -> bytes:
        return encode_line(
            {
                "type": "user",
                "message": {"role": "user", "content": text},
                "parent_tool_use_id": None,
                "session_id": "default",
            }
        )

    def _control(self, request_id: str, request: Mapping[str, object]) -> bytes:
        return encode_line(
            {"type": "control_request", "request_id": request_id, "request": request}
        )

    def submit(self, command: SessionInput) -> ProtocolStep:
        if not self._started or not self._active:
            raise ValueError("ClaudeSession is not active")
        if command.action == "steer":
            raise ValueError("Claude does not support native steer; use follow_up")
        if command.action == "follow_up":
            return self._follow_up(command)
        if command.action == "interrupt":
            return self._interrupt()
        if command.action in ("approve", "deny"):
            return self._permission(command)
        raise ValueError(f"Unsupported Claude session action: {command.action}")

    def _follow_up(self, command: SessionInput) -> ProtocolStep:
        if not command.text:
            raise ValueError("Claude follow_up requires text")
        request_id = command.request_id or self._id("user")
        if request_id in self._user_ids:
            raise ValueError(f"Duplicate Claude user request_id: {request_id}")
        turn = Turn(request_id, command.text)
        self._user_ids.add(request_id)
        self._turns.append(turn)
        self._pending_users.append(turn)
        return ProtocolStep(commands=(self._user(command.text),))

    def _interrupt(self) -> ProtocolStep:
        if self._interrupt_requested:
            raise ValueError("Claude interrupt is already pending")
        request_id = self._id("interrupt")
        self._controls[request_id] = "interrupt"
        self._interrupt_requested = True
        event = SessionEvent(
            kind="status", text="Interrupt requested", event_id=request_id, state="running"
        )
        return ProtocolStep(
            commands=(self._control(request_id, {"subtype": "interrupt"}),), events=(event,)
        )

    def _permission(self, command: SessionInput) -> ProtocolStep:
        request_id = command.request_id
        permission = self._permissions.pop(request_id, None)
        if not request_id or permission is None:
            raise ValueError(f"Unknown Claude permission request_id: {request_id!r}")
        if command.action == "approve":
            response = {"behavior": "allow", "updatedInput": permission.input}
            state, text = "approved", f"Permission approved: {permission.text}"
        else:
            response = {"behavior": "deny", "message": command.text or "Denied by operator."}
            state, text = "denied", f"Permission denied: {permission.text}"
        wire = {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": response,
            },
        }
        submitted = SessionEvent(
            kind="permission", text=text, event_id=request_id, state="submitted"
        )
        decision = SessionEvent(kind="permission", text=text, event_id=request_id, state=state)
        return ProtocolStep(
            commands=(encode_line(wire),),
            events=(submitted,),
            after_write_events=(decision,),
        )

    def receive(self, line: bytes) -> ProtocolStep:
        if not self._started:
            raise ValueError("ClaudeSession has not started")
        self._buffer.extend(line)
        commands: list[bytes] = []
        events: list[SessionEvent] = []
        after_write_events: list[SessionEvent] = []
        done = failed = interrupted = False
        result_text: str | None = None
        session_id = self._session_id
        for frame in self._frames():
            step = self._frame(frame)
            commands.extend(step.commands)
            events.extend(step.events)
            after_write_events.extend(step.after_write_events)
            done |= step.done
            failed |= step.failed
            interrupted |= step.interrupted
            result_text = step.result_text if step.result_text is not None else result_text
            session_id = step.session_id or session_id
        self._session_id = session_id
        return ProtocolStep(
            commands=tuple(commands),
            events=tuple(events),
            after_write_events=tuple(after_write_events),
            done=done,
            result_text=result_text,
            failed=failed,
            interrupted=interrupted,
            session_id=session_id,
        )

    def _frames(self) -> list[bytes]:
        if b"\n" not in self._buffer:
            return []
        parts = bytes(self._buffer).split(b"\n")
        self._buffer = bytearray(parts.pop())
        return [part for part in parts if part.strip()]

    def _frame(self, frame: bytes) -> ProtocolStep:
        try:
            raw = msgspec.json.decode(frame, type=ClaudeFrame)
        except (msgspec.DecodeError, msgspec.ValidationError, TypeError) as exc:
            return self._error(f"Invalid Claude stream-json frame: {exc}")
        kind = raw.type
        if kind == "assistant":
            step = self._assistant(raw)
        elif kind == "stream_event":
            step = self._stream(raw)
        elif kind == "user":
            step = self._user_echo(raw)
        elif kind == "result":
            step = self._result(raw)
        elif kind == "control_request":
            step = self._control_request(raw)
        elif kind == "control_response":
            step = self._control_response(raw)
        elif kind == "control_cancel_request":
            step = self._control_cancel(raw)
        elif kind == "error":
            step = self._error(self._error_text(raw))
        elif kind == "system":
            step = self._system(raw)
        else:
            step = ProtocolStep(
                events=(
                    SessionEvent(
                        kind="status",
                        text=f"Claude event: {kind}",
                        event_id=self._frame_id(raw, kind),
                        state="running",
                    ),
                )
            )
        session_id = self._frame_session_id(raw) or step.session_id
        if session_id == step.session_id:
            return step
        return ProtocolStep(
            commands=step.commands,
            events=step.events,
            done=step.done,
            result_text=step.result_text,
            failed=step.failed,
            interrupted=step.interrupted,
            session_id=session_id,
        )

    def _frame_session_id(self, raw: ClaudeFrame) -> str | None:
        return raw.session_id
