from __future__ import annotations

from typing import cast

from milknado.domains.common.session import SessionEvent
from milknado.loop.sessions._protocol import ProtocolStep

from ._claude_state import (
    Block,
    ClaudeContentBlock,
    ClaudeFrame,
    ClaudeMessage,
    ClaudeState,
    ClaudeStreamEvent,
)


def _mapping(value: object) -> dict[str, object] | None:
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


class ClaudeEventsMixin(ClaudeState):
    _assistant_id: str | None

    def _assistant(self, raw: ClaudeFrame) -> ProtocolStep:
        message = raw.message
        if not isinstance(message, ClaudeMessage):
            return self._error("Claude assistant frame is missing message")
        event_id = message.id or raw.uuid or self._id("assistant")
        blocks = message.content
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        if not isinstance(blocks, list):
            return self._error("Claude assistant frame is missing content")
        text_parts: list[str] = []
        events: list[SessionEvent] = []
        for block in blocks:
            data = _mapping(block) or {}
            kind = _string(data.get("type"))
            if kind in ("text", "thinking"):
                text_parts.append(_string(data.get("text")) or _string(data.get("thinking")) or "")
            elif kind == "tool_use":
                events.append(self._tool(data, "streaming"))
            elif kind == "tool_result":
                events.append(self._tool(data, "complete"))
        if text_parts:
            text = "".join(text_parts)
            self._assistant_text[event_id] = text
            events.insert(
                0,
                SessionEvent(kind="assistant", text=text, event_id=event_id, state="complete"),
            )
        return ProtocolStep(events=tuple(events), session_id=raw.session_id)

    def _tool(self, data: dict[str, object], state: str) -> SessionEvent:
        event_id = _string(data.get("id")) or _string(data.get("tool_use_id")) or self._id("tool")
        name = _string(data.get("name")) or "tool"
        value = _mapping(data.get("input")) or {}
        detail = _string(value.get("command")) or _string(value.get("file_path"))
        text = (
            f"{name}: {detail}"
            if detail
            else (self._content(data.get("content")) if state == "complete" else name)
        )
        return SessionEvent(kind="tool", text=text, event_id=event_id, state=state)

    def _stream(self, raw: ClaudeFrame) -> ProtocolStep:
        event = raw.event
        if event is None:
            return ProtocolStep(session_id=raw.session_id)
        if event.type == "message_start":
            message = event.message
            self._assistant_id = (
                (message.id if message else None) or raw.uuid or self._id("assistant")
            )
            _ = self._assistant_text.setdefault(self._assistant_id, "")
            self._blocks.clear()
        elif event.type == "content_block_start":
            return self._block_start(event, raw.session_id)
        elif event.type == "content_block_delta":
            return self._block_delta(event, raw.session_id)
        elif event.type == "message_stop":
            return self._message_stop(raw.session_id)
        return ProtocolStep(session_id=raw.session_id)

    def _block_start(self, event: ClaudeStreamEvent, session_id: str | None) -> ProtocolStep:
        block = event.content_block or ClaudeContentBlock()
        if event.index is None:
            return ProtocolStep(session_id=session_id)
        event_id = self._assistant_id or self._id("assistant")
        if block.type == "tool_use":
            event_id = block.id or self._id("tool")
            self._blocks[event.index] = Block("tool", event_id, block.name or "tool")
            output = SessionEvent(
                kind="tool", text=block.name or "tool", event_id=event_id, state="streaming"
            )
        else:
            self._blocks[event.index] = Block(block.type, event_id)
            output = SessionEvent(kind="assistant", text="", event_id=event_id, state="streaming")
        return ProtocolStep(events=(output,), session_id=session_id)

    def _block_delta(self, event: ClaudeStreamEvent, session_id: str | None) -> ProtocolStep:
        if event.index is None or event.delta is None:
            return ProtocolStep(session_id=session_id)
        block = self._blocks.get(event.index)
        delta = event.delta
        value = delta.text or delta.thinking
        if block is None or value is None or delta.type not in ("text_delta", "thinking_delta"):
            return ProtocolStep(session_id=session_id)
        self._assistant_text[block.event_id] = self._assistant_text.get(block.event_id, "") + value
        output = SessionEvent(
            kind="assistant", text=value, event_id=block.event_id, state="streaming", delta=True
        )
        return ProtocolStep(events=(output,), session_id=session_id)

    def _message_stop(self, session_id: str | None) -> ProtocolStep:
        events: tuple[SessionEvent, ...] = ()
        if self._assistant_id and self._assistant_text.get(self._assistant_id):
            events = (
                SessionEvent(
                    kind="assistant",
                    text=self._assistant_text[self._assistant_id],
                    event_id=self._assistant_id,
                    state="complete",
                ),
            )
        self._blocks.clear()
        self._assistant_id = None
        return ProtocolStep(events=events, session_id=session_id)

    def _user_echo(self, raw: ClaudeFrame) -> ProtocolStep:
        message = raw.message
        if isinstance(message, str):
            return ProtocolStep(events=(self._delivered(message, raw),), session_id=raw.session_id)
        if not isinstance(message, ClaudeMessage):
            return self._error("Claude user frame is missing message")
        content = message.content
        if isinstance(content, list):
            events: list[SessionEvent] = []
            text_parts: list[str] = []
            for block in content:
                data = _mapping(block) or {}
                if _string(data.get("type")) == "tool_result":
                    events.append(self._tool(data, "complete"))
                elif _string(data.get("type")) == "text":
                    text_parts.append(_string(data.get("text")) or "")
            if text_parts:
                events.append(self._delivered("".join(text_parts), raw))
            return ProtocolStep(events=tuple(events), session_id=raw.session_id)
        return ProtocolStep(
            events=(self._delivered(content or "", raw),), session_id=raw.session_id
        )

    def _delivered(self, text: str, raw: ClaudeFrame) -> SessionEvent:
        event_id = raw.uuid or self._id("user")
        for turn in self._pending_users:
            if turn.text == text:
                event_id = turn.request_id
                self._pending_users.remove(turn)
                break
        return SessionEvent(kind="user", text=text, event_id=event_id, state="delivered")

    def _content(self, value: object) -> str:
        if isinstance(value, str):
            return value
        if not isinstance(value, list):
            return ""
        parts: list[str] = []
        for item in cast(list[object], value):
            data = _mapping(item)
            if data:
                parts.append(_string(data.get("text")) or "")
        return "".join(parts)
