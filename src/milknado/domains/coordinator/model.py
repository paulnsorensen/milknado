from __future__ import annotations

from dataclasses import dataclass

import msgspec


@dataclass(frozen=True)
class CoordinatorSession:
    id: str
    goal_id: int
    provider: str
    created_at: str


@dataclass(frozen=True)
class EntityLink:
    kind: str
    entity_id: str


class ControlEvent(msgspec.Struct, frozen=True):
    kind: str
    text: str = ""
    entity_kind: str = ""
    entity_id: str = ""
    tool_name: str = ""
    status: str = ""
    duration_ms: int | None = None
    tool_arguments: str = ""  # noqa: V107 - accepted but never persisted
    tool_result: str = ""  # noqa: V107 - accepted but never persisted
    diagnostic_retention_seconds: int | None = None


@dataclass(frozen=True)
class ControlRecord:
    seq: int
    kind: str
    text: str
    entity_kind: str
    entity_id: str
    tool_name: str
    status: str
    duration_ms: int | None
    created_at: str
