from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from milknado.domains.common import SessionAction, SessionEvent, SessionInput


@dataclass(frozen=True, slots=True)
class ProtocolStep:
    commands: tuple[bytes, ...] = ()
    events: tuple[SessionEvent, ...] = ()
    after_write_events: tuple[SessionEvent, ...] = ()
    done: bool = False
    result_text: str | None = None
    failed: bool = False
    interrupted: bool = False
    session_id: str | None = None


class SessionProtocol(Protocol):
    command: tuple[str, ...]

    @property
    def actions(self) -> tuple[SessionAction, ...]: ...

    def start(self, prompt: str) -> ProtocolStep: ...
    def receive(self, line: bytes) -> ProtocolStep: ...
    def submit(self, command: SessionInput) -> ProtocolStep: ...
