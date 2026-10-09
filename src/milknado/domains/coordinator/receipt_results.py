from __future__ import annotations

import sqlite3
from typing import Literal, cast

import msgspec

from milknado.domains.coordinator.control_models import CoordinatorCommandReceipt
from milknado.domains.coordinator.recovery import CoordinatorRecovery


class RecoveryItem(msgspec.Struct, frozen=True):
    entity_kind: str
    entity_id: str
    provider_family: str | None  # noqa: V107
    provider_session_id: str | None
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


def reserve_command_receipt(
    conn: sqlite3.Connection, session_id: str, command_id: str, fingerprint: str
) -> CoordinatorCommandReceipt | None:
    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO coordinator_web_receipts "
            + "(command_id, session_id, command_hash, status, result_json) "
            + "VALUES (?, ?, ?, 'unconfirmed', 'null')",
            (command_id, session_id, fingerprint),
        )
    row = cast(
        tuple[str, str, str, str] | None,
        conn.execute(
            "SELECT session_id, command_hash, status, result_json "
            + "FROM coordinator_web_receipts WHERE command_id = ?",
            (command_id,),
        ).fetchone(),
    )
    if row is None:
        raise RuntimeError("coordinator command has no receipt")
    if row[:2] != (session_id, fingerprint):
        raise ValueError("command_id was reused for a different command")
    if cursor.rowcount:
        return None
    return CoordinatorCommandReceipt(
        command_id,
        session_id,
        cast(
            Literal["accepted", "unavailable", "unsupported", "rejected", "unconfirmed"],
            row[2],
        ),
        cast(object, msgspec.json.decode(row[3].encode())),
    )


def receipt_payload(result: object) -> object:
    match result:
        case CoordinatorRecovery():
            payload = RecoveryCommandResult(
                result.session.id,
                tuple(
                    RecoveryItem(
                        item.entity_kind,
                        item.entity_id,
                        item.identity.family if item.identity else None,
                        item.identity.session_id if item.identity else None,
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
