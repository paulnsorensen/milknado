from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import msgspec

from milknado.domains.coordinator.recovery import CoordinatorRecovery


@dataclass(frozen=True, slots=True)
class RecoveryItem:
    entity_kind: str
    entity_id: str
    provider_family: str  # noqa: V107
    provider_session_id: str
    worktree_path: str
    outcome: str


@dataclass(frozen=True, slots=True)
class UnknownTurnItem:
    provider_family: str  # noqa: V107
    provider_session_id: str
    turn_id: str


@dataclass(frozen=True, slots=True)
class RecoveryCommandResult:
    session_id: str
    receipts: tuple[RecoveryItem, ...]
    unknown_turns: tuple[UnknownTurnItem, ...]


def receipt_payload(result: object) -> object:
    match result:
        case CoordinatorRecovery():
            payload = RecoveryCommandResult(
                result.session.id,
                tuple(
                    RecoveryItem(
                        item.entity_kind,
                        item.entity_id,
                        item.identity.family,
                        item.identity.session_id,
                        str(item.worktree_path),
                        item.outcome,
                    )
                    for item in result.receipts
                ),
                tuple(
                    UnknownTurnItem(item.identity.family, item.identity.session_id, item.turn_id)
                    for item in result.unknown_turns
                ),
            )
        case _:
            payload = result
    return cast(object, msgspec.json.decode(msgspec.json.encode(msgspec.to_builtins(payload))))
