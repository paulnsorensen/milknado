from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import msgspec

from milknado.domains.coordinator.recovery import CoordinatorRecovery
from milknado.domains.planning import PlanResult


@dataclass(frozen=True, slots=True)
class PlanCommandResult:
    success: bool
    exit_code: int
    context_path: str | None
    nodes_created: int
    batch_count: int
    oversized_count: int
    solver_status: str
    change_count: int
    mega_batch_change_count: int | None


@dataclass(frozen=True, slots=True)
class RecoveryItem:
    entity_kind: str
    entity_id: str
    provider_family: str
    provider_session_id: str
    worktree_path: str
    outcome: str


@dataclass(frozen=True, slots=True)
class UnknownTurnItem:
    provider_family: str
    provider_session_id: str
    turn_id: str


@dataclass(frozen=True, slots=True)
class RecoveryCommandResult:
    session_id: str
    receipts: tuple[RecoveryItem, ...]
    unknown_turns: tuple[UnknownTurnItem, ...]


def receipt_payload(result: object) -> object:
    match result:
        case PlanResult():
            payload = PlanCommandResult(
                result.success,
                result.exit_code,
                str(result.context_path) if result.context_path is not None else None,
                result.nodes_created,
                result.batch_count,
                result.oversized_count,
                result.solver_status,
                result.change_count,
                result.mega_batch_change_count,
            )
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
