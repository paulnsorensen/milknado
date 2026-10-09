from __future__ import annotations

from typing import cast

import msgspec

from milknado.domains.coordinator.recovery import CoordinatorRecovery
from milknado.domains.planning import PlanResult


class PlanCommandResult(msgspec.Struct, frozen=True):
    success: bool
    exit_code: int
    context_path: str | None
    nodes_created: int
    batch_count: int
    oversized_count: int
    solver_status: str
    change_count: int
    mega_batch_change_count: int | None


class RecoveryItem(msgspec.Struct, frozen=True):
    entity_kind: str
    entity_id: str
    provider_family: str  # noqa: V107
    provider_session_id: str
    worktree_path: str
    outcome: str


class UnknownTurnItem(msgspec.Struct, frozen=True):
    provider_family: str  # noqa: V107
    provider_session_id: str
    turn_id: str


class RecoveryCommandResult(msgspec.Struct, frozen=True):
    session_id: str
    receipts: tuple[RecoveryItem, ...]
    unknown_turns: tuple[UnknownTurnItem, ...]


def receipt_payload(result: object) -> object:
    payload: object = result
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
