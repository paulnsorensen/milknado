from __future__ import annotations

from milknado.domains.common import SessionEvent, SessionInput
from milknado.loop.sessions._omp_events import OmpEventMixin
from milknado.loop.sessions._omp_state import Pending, UiRequest
from milknado.loop.sessions._omp_wire import OmpFrame, OmpResponse, encode, text
from milknado.loop.sessions._protocol import ProtocolStep


class OmpControlMixin(OmpEventMixin):
    _session_id: str | None
    _interrupt_accepted: bool
    _active: bool

    def submit(self, command: SessionInput) -> ProtocolStep:
        if not self._active:
            raise ValueError("OMP session is inactive")
        if command.action in {"steer", "follow_up", "interrupt"}:
            return self._submit_control(command)
        if command.action in {"approve", "deny"}:
            return self._submit_ui(command)
        raise ValueError(f"Unsupported OMP session action: {command.action}")

    def _submit_control(self, command: SessionInput) -> ProtocolStep:
        request_id = command.request_id or self._next_id(command.action)
        if request_id in self._pending:
            raise ValueError(f"OMP request ID is already pending: {request_id}")
        self._pending[request_id] = Pending(command.action, command.text)
        if command.action == "interrupt":
            payload: dict[str, object] = {"id": request_id, "type": "abort"}
            event = SessionEvent(
                kind="status", text="interrupt submitted", event_id=request_id, state="submitted"
            )
            return ProtocolStep(commands=(encode(payload),), events=(event,))
        payload = {"id": request_id, "type": command.action, "message": command.text}
        return ProtocolStep(commands=(encode(payload),))

    def _submit_ui(self, command: SessionInput) -> ProtocolStep:
        request_id = command.request_id
        request = self._ui_requests.get(request_id)
        if request is None:
            raise ValueError(f"Unknown or stale OMP UI request: {request_id}")
        if request.method == "select" and command.action == "approve":
            if command.text not in request.options:
                raise ValueError(f"OMP select reply must match one of {request.options!r}")
            payload: dict[str, object] = {
                "type": "extension_ui_response",
                "id": request_id,
                "value": command.text,
            }
        elif request.method in {"input", "editor"} and command.action == "approve":
            payload = {"type": "extension_ui_response", "id": request_id, "value": command.text}
        elif request.method == "confirm":
            payload = {
                "type": "extension_ui_response",
                "id": request_id,
                "confirmed": command.action == "approve",
            }
        else:
            payload = {"type": "extension_ui_response", "id": request_id, "cancelled": True}
        del self._ui_requests[request_id]
        event = SessionEvent(
            kind="permission", text=request.title, event_id=request_id, state="submitted"
        )
        return ProtocolStep(commands=(encode(payload),), events=(event,))

    def _response(self, frame: OmpFrame) -> ProtocolStep:
        request_id = frame.id
        command_name = frame.command
        if not isinstance(request_id, str) or not isinstance(command_name, str):
            return ProtocolStep(
                events=(
                    self._error(
                        "OMP response is missing correlation id or command", "stale_response"
                    ),
                )
            )
        pending = self._pending.pop(request_id, None)
        if pending is None:
            state = "resolved_response" if request_id in self._resolved else "stale_response"
            return ProtocolStep(
                events=(self._error(f"OMP response for unknown request {request_id}", state),)
            )
        expected = "abort" if pending.action == "interrupt" else pending.action
        if command_name != expected:
            self._resolved.add(request_id)
            return ProtocolStep(
                events=(
                    self._error(
                        f"OMP response command mismatch: expected {expected}, got {command_name}",
                        "response_mismatch",
                    ),
                )
            )
        self._resolved.add(request_id)
        if pending.action in {"get_state", "negotiate_protocol"}:
            if frame.success is not True:
                message = (
                    text(frame.error) or text(frame.message) or f"OMP rejected {command_name}"
                )
                return ProtocolStep(events=(self._error(message, "rejected", request_id),))
            if pending.action == "get_state":
                data = frame.data if isinstance(frame.data, OmpResponse) else None
                session_id = data.session_id if data is not None else None
                if isinstance(session_id, str) and session_id:
                    self._session_id = session_id
            state = "ready" if pending.action == "get_state" else "negotiated"
            return ProtocolStep(
                events=(self._status(state, f"OMP {pending.action} acknowledged", request_id),),
                session_id=self._session_id,
            )
        if frame.success is not True:
            return self._reject_pending(
                request_id,
                pending,
                text(frame.error) or text(frame.message) or f"OMP rejected {command_name}",
            )
        if pending.action in {"steer", "follow_up"}:
            if not pending.acknowledged:
                self._queued[request_id] = (pending.action, pending.text)
            return ProtocolStep()
        if pending.action == "interrupt":
            self._interrupt_accepted = True
            event = SessionEvent(
                kind="status", text="interrupt accepted", event_id=request_id, state="queued"
            )
        else:
            data = frame.data if isinstance(frame.data, OmpResponse) else None
            if data is not None and data.agent_invoked is False:
                self._active = False
                return ProtocolStep(
                    done=True,
                    result_text=self._result_text,
                    failed=self._failed,
                    session_id=self._session_id,
                )
            if not pending.acknowledged:
                self._queued[request_id] = (pending.action, pending.text)
            return ProtocolStep()
        if pending.acknowledged:
            return ProtocolStep()
        return ProtocolStep(events=(event,))

    def _reject_pending(self, request_id: str, pending: Pending, text: str) -> ProtocolStep:
        kind = "status" if pending.action == "interrupt" else "user"
        event = SessionEvent(kind=kind, text=text, event_id=request_id, state="rejected")
        return ProtocolStep(events=(event, self._error(text, "rejected", request_id)), failed=True)

    def _reject(self, frame: OmpFrame, text: str, state: str) -> ProtocolStep:
        request_id = frame.id
        if isinstance(request_id, str):
            pending = self._pending.pop(request_id, None)
            if pending is not None:
                self._resolved.add(request_id)
                return self._reject_pending(request_id, pending, text)
        return ProtocolStep(
            events=(
                self._error(text, state, request_id if isinstance(request_id, str) else None),
            ),
            failed=True,
        )

    def _ui_request(self, frame: OmpFrame) -> ProtocolStep:
        request_id = frame.id
        method = frame.method
        if not isinstance(request_id, str) or not isinstance(method, str):
            return ProtocolStep(
                events=(
                    self._error("OMP UI request is missing id or method", "invalid_ui_request"),
                )
            )
        title = text(frame.title)
        if method in self._INTERACTIVE_UI:
            options_value = frame.options
            options = tuple(options_value) if options_value is not None else ()
            self._ui_requests[request_id] = UiRequest(method, title, options)
            detail = (
                f"{title}: {', '.join(options)}"
                if method == "select"
                else title or text(frame.message)
            )
            return ProtocolStep(
                events=(
                    SessionEvent(
                        kind="permission", text=detail, event_id=request_id, state="requested"
                    ),
                )
            )
        if method in self._PASSIVE_UI:
            detail = text(frame.message) or title or method
            return ProtocolStep(
                events=(
                    SessionEvent(kind="status", text=detail, event_id=request_id, state=method),
                )
            )
        return ProtocolStep(
            events=(
                self._error(
                    f"Unsupported OMP extension UI request: {method}", "unsupported_ui", request_id
                ),
            )
        )
