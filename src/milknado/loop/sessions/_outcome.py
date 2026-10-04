"""Mutable outcome for one native agent session."""

from __future__ import annotations

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

    @property
    def graceful(self) -> bool:
        return not (self.timed_out or self.force_stopped or not self.done or self.capped)


def publish_events(channel: SessionChannel, events: tuple[SessionEvent, ...]) -> None:
    _ = tuple(map(channel.publish, events))
