from __future__ import annotations

from abc import ABCMeta

from typing_extensions import override

from milknado.domains.common import SessionEvent, SessionInput
from milknado.loop.sessions._codex_events import object_value, text_value
from milknado.loop.sessions._codex_types import ApprovalRequest, CodexFrame, CodexState
from milknado.loop.sessions._protocol import ProtocolStep


def approval_text(method: str, params: dict[str, object]) -> str:
    if method in {"item/commandExecution/requestApproval", "execCommandApproval"}:
        return text_value(params.get("command")) or "Codex requests command execution approval"
    if method in {"item/fileChange/requestApproval", "applyPatchApproval"}:
        return text_value(params.get("reason")) or "Codex requests file change approval"
    if method == "item/permissions/requestApproval":
        return text_value(params.get("reason")) or "Codex requests additional permissions"
    return text_value(params.get("message")) or "Codex requests user input"


def approval_result(approval: ApprovalRequest, command: SessionInput) -> dict[str, object]:
    method = approval.method
    if method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
        allowed = {"approve": {"accept", "acceptForSession"}, "deny": {"decline", "cancel"}}[
            command.action
        ]
        choice = command.text.strip() or ("accept" if command.action == "approve" else "decline")
        if choice not in allowed:
            raise ValueError(f"unsupported Codex approval choice: {choice!r}")
        return {"decision": choice}
    if method == "item/permissions/requestApproval":
        requested = object_value(approval.params.get("permissions"))
        permissions = (
            requested if command.action == "approve" else {"fileSystem": None, "network": None}
        )
        result: dict[str, object] = {"permissions": permissions}
        if command.text.strip().lower() == "session":
            result["scope"] = "session"
        return result
    if method in {"execCommandApproval", "applyPatchApproval"}:
        return {"decision": "approved" if command.action == "approve" else "denied"}
    if method == "tool/requestUserInput":
        return {"answers": {}}
    if method == "mcpServer/elicitation/request":
        return {"action": "accept" if command.action == "approve" else "decline", "content": None}
    raise ValueError(f"unsupported Codex approval request: {method}")


class CodexApprovalMixin(CodexState, metaclass=ABCMeta):
    @override
    def _approval_request(self, frame: CodexFrame, params: dict[str, object]) -> ProtocolStep:
        raw_id = frame.id
        if not isinstance(raw_id, (int, str)) or isinstance(raw_id, bool):
            return self._failure("Codex approval request id must be a string or integer")
        request_id = str(raw_id)
        if request_id in self._approvals:
            return self._failure(f"duplicate Codex approval request id {request_id!r}")
        method = frame.method or ""
        self._approvals[request_id] = ApprovalRequest(raw_id, method, params)
        event = SessionEvent(
            kind="permission",
            text=approval_text(method, params),
            event_id=request_id,
            state="requested",
        )
        return ProtocolStep(events=(event,), session_id=self._session_id or None)

    @override
    def _approval_resolved(self, params: dict[str, object]) -> ProtocolStep:
        request_id = text_value(params.get("requestId") or params.get("id"))
        approval = self._approvals.pop(request_id, None)
        if approval is None:
            return ProtocolStep()
        state = "approved" if approval.action == "approve" else "denied"
        event = SessionEvent(
            kind="permission",
            text=approval_text(approval.method, approval.params),
            event_id=request_id,
            state=state,
        )
        return ProtocolStep(events=(event,), session_id=self._session_id or None)
