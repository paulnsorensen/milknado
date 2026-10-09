from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import msgspec


@dataclass(frozen=True)
class CoordinatorSession:
    id: str
    goal_id: int
    provider: str
    created_at: str


class CoordinatorSessionSummary(msgspec.Struct, frozen=True):
    id: str
    goal_id: int
    provider: str
    created_at: str
    description: str


@dataclass(frozen=True)
class EntityLink:
    kind: str
    entity_id: str


@dataclass(frozen=True)
class ProviderBinding:
    scope_kind: str
    scope_id: str
    family: str
    provider_session_id: str


RecoveryOutcome = Literal["reattached", "resumed", "unknown_turn", "unavailable", "unsupported"]


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    family: str
    session_id: str

    def __post_init__(self) -> None:
        if self.family not in {"claude", "codex"} or not self.session_id:
            raise ValueError("recovery requires a supported provider session identity")


@dataclass(frozen=True, slots=True)
class RecoveryReceipt:
    entity_kind: str
    entity_id: str
    identity: ProviderIdentity | None
    worktree_path: Path
    outcome: RecoveryOutcome


class ControlEvent(msgspec.Struct, frozen=True):
    kind: str
    text: str = ""
    entity_kind: str = ""
    entity_id: str = ""
    tool_name: str = ""
    status: str = ""
    turn_id: str = ""
    provider_session_id: str = ""
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
    turn_id: str
    provider_session_id: str
    duration_ms: int | None
    created_at: str
