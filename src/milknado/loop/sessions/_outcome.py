"""Mutable outcome for one native agent session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from milknado.domains.common import SessionEvent
from milknado.loop.sessions._channel import SessionChannel


@dataclass(slots=True)
class SessionOutcome:
    done: bool = False
    failed: bool = False
    interrupted: bool = False
    result_text: str | None = None
    session_id: str | None = None
    tool_count: int = 0
    capped: bool = False
    timed_out: bool = False
    force_stopped: bool = False
    interrupt_requested: bool = False
    tool_ids: set[str] = field(default_factory=set)
    reader_failed: bool = False

    def remember_session(
        self, session_id: str | None, on_session_id: Callable[[str], None] | None
    ) -> None:
        if session_id is None:
            return
        if self.session_id is None and on_session_id is not None:
            on_session_id(session_id)
        self.session_id = session_id

    @property
    def graceful(self) -> bool:
        return not (self.timed_out or self.force_stopped or not self.done or self.capped)

    @property
    def terminal_confirmed(self) -> bool:
        return self.graceful and not self.failed


def publish_events(channel: SessionChannel, events: tuple[SessionEvent, ...]) -> None:
    _ = tuple(map(channel.publish, events))
