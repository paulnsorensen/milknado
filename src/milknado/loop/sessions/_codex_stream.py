from __future__ import annotations

from abc import ABCMeta
from typing import cast

from typing_extensions import override

from milknado.domains.common import SessionEvent
from milknado.loop.sessions._codex_events import error_text, object_value, text_value
from milknado.loop.sessions._codex_types import CodexDeltaKind, CodexState
from milknado.loop.sessions._protocol import ProtocolStep


def item_kind(item: dict[str, object]) -> str:
    return text_value(item.get("type"))


def item_id(item: dict[str, object], params: dict[str, object]) -> str:
    return text_value(item.get("id") or params.get("itemId"))


def item_text(item: dict[str, object]) -> str:
    kind = item_kind(item)
    if kind in {"agentMessage", "plan"}:
        return text_value(item.get("text"))
    if kind == "userMessage":
        content = item.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for part in cast(list[object], content):
                data = object_value(part)
                if data.get("type") == "text":
                    parts.append(text_value(data.get("text")))
            return "\n".join(parts)
        return text_value(content)
    if kind == "commandExecution":
        return text_value(item.get("aggregatedOutput")) or text_value(item.get("command"))
    if kind == "mcpToolCall":
        return text_value(item.get("name")) or text_value(item.get("tool"))
    if kind == "fileChange":
        return ", ".join(str(path) for path in object_value(item.get("changes")))
    if kind == "functionCallOutput":
        return text_value(item.get("output")) or text_value(item.get("name"))
    if kind in {"dynamicToolCall", "collabAgentToolCall", "subAgentActivity"}:
        return text_value(item.get("name")) or text_value(item.get("tool")) or text_value(item)
    return ""


def is_tool(kind: str) -> bool:
    return kind in {
        "commandExecution",
        "mcpToolCall",
        "fileChange",
        "functionCallOutput",
        "dynamicToolCall",
        "collabAgentToolCall",
        "subAgentActivity",
    }


class CodexStreamMixin(CodexState, metaclass=ABCMeta):
    _turn_id: str
    _active: bool
    _fatal: bool

    @override
    def _delta_event(self, params: dict[str, object], kind: CodexDeltaKind) -> ProtocolStep:
        event_id = text_value(params.get("itemId"))
        if not event_id:
            return self._failure("Codex delta omitted itemId")
        delta = text_value(params.get("delta"))
        if kind == "assistant":
            self._assistant[event_id] = self._assistant.get(event_id, "") + delta
        event = SessionEvent(
            kind=kind, text=delta, event_id=event_id, state="streaming", delta=True
        )
        return ProtocolStep(events=(event,), session_id=self._session_id or None)

    @override
    def _progress_event(self, params: dict[str, object]) -> ProtocolStep:
        event_id = text_value(params.get("itemId"))
        if not event_id:
            return self._failure("Codex tool progress omitted itemId")
        event = SessionEvent(
            kind="tool",
            text=text_value(params.get("message")),
            event_id=event_id,
            state="streaming",
            delta=True,
        )
        return ProtocolStep(events=(event,), session_id=self._session_id or None)

    @override
    def _terminal_event(self, params: dict[str, object]) -> ProtocolStep:
        event_id = text_value(params.get("itemId"))
        if not event_id:
            return self._failure("Codex terminal event omitted itemId")
        event = SessionEvent(
            kind="tool",
            text=text_value(params.get("stdin")),
            event_id=event_id,
            state="streaming",
            delta=True,
        )
        return ProtocolStep(events=(event,), session_id=self._session_id or None)

    @override
    def _item_event(self, params: dict[str, object], complete: bool) -> ProtocolStep:
        item = object_value(params.get("item"))
        kind = item_kind(item)
        event_id = item_id(item, params)
        if not event_id:
            return self._failure("Codex item event omitted item id")
        if not kind:
            return self._failure("Codex item event omitted item type")
        text = item_text(item)
        state = "complete" if complete else "streaming"
        if kind == "agentMessage":
            self._assistant[event_id] = text or self._assistant.get(event_id, "")
            event = SessionEvent(
                kind="assistant",
                text=text or self._assistant[event_id],
                event_id=event_id,
                state=state,
            )
        elif kind == "plan":
            event = SessionEvent(kind="assistant", text=text, event_id=event_id, state=state)
        elif is_tool(kind):
            event = SessionEvent(kind="tool", text=text, event_id=event_id, state=state)
        elif kind == "userMessage":
            event = SessionEvent(
                kind="user",
                text=text,
                event_id=text_value(item.get("clientId")) or event_id,
                state="delivered",
            )
        else:
            return ProtocolStep(session_id=self._session_id or None)
        return ProtocolStep(events=(event,), session_id=self._session_id or None)

    @override
    def _result(self, turn: dict[str, object]) -> str | None:
        items = turn.get("items")
        if isinstance(items, list):
            for item in reversed(cast(list[object], items)):
                data = object_value(item)
                if item_kind(data) == "agentMessage":
                    text = item_text(data)
                    if text:
                        return text
        return next(reversed(self._assistant.values()), None) if self._assistant else None

    @override
    def _turn_completed(self, params: dict[str, object]) -> ProtocolStep:
        turn = object_value(params.get("turn"))
        turn_id = text_value(turn.get("id")) or self._turn_id
        if self._turn_id and turn_id and turn_id != self._turn_id:
            return ProtocolStep(session_id=self._session_id or None)
        self._turn_id = turn_id
        status = text_value(turn.get("status")) or "completed"
        self._active = False
        failed = status == "failed" or bool(turn.get("error"))
        interrupted = status == "interrupted"
        events: list[SessionEvent] = [
            SessionEvent(
                kind="status",
                text=f"turn {status}",
                event_id=turn_id,
                state="stopped" if interrupted or failed else "complete",
            )
        ]
        if turn.get("error"):
            events.append(
                SessionEvent(
                    kind="error",
                    text=error_text(turn["error"]),
                    event_id=turn_id,
                    state="rejected",
                )
            )
        return ProtocolStep(
            events=tuple(events),
            done=True,
            result_text=self._result(turn),
            failed=failed,
            interrupted=interrupted,
            session_id=self._session_id or None,
        )

    @override
    def _turn_error(self, params: dict[str, object]) -> ProtocolStep:
        turn_id = text_value(params.get("turnId")) or self._turn_id
        event = SessionEvent(
            kind="error", text=error_text(params.get("error")), event_id=turn_id, state="rejected"
        )
        if params.get("willRetry") is True:
            return ProtocolStep(events=(event,), session_id=self._session_id or None)
        self._active = False
        self._fatal = True
        return ProtocolStep(
            events=(event,), done=True, failed=True, session_id=self._session_id or None
        )
