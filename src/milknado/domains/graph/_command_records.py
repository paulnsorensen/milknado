"""Hydration and validation helpers for durable graph commands."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import cast

from milknado.domains.common.session import SessionAction
from milknado.domains.graph._sqlite_rows import fetchall, fetchone
from milknado.domains.graph.commands import (
    CommandReceipt,
    CommandStatus,
    GraphCommand,
    OwnerCapabilities,
)

_MAX_PENDING_COMMANDS = 64
_MAX_COMMAND_TEXT_BYTES = 64 * 1024
_MAX_IDENTIFIER_BYTES = 256
_ACTIONS = frozenset({"steer", "follow_up", "interrupt", "approve", "deny"})
_TERMINAL = frozenset({"delivered", "rejected", "expired", "unconfirmed"})
_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"submitted", "rejected", "unconfirmed"}),
    "submitted": frozenset({"delivered", "rejected", "unconfirmed"}),
}


def utc_iso(value: str | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"timestamp is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def validate_identifier(value: str, name: str) -> None:
    if not value or len(value.encode()) > _MAX_IDENTIFIER_BYTES:
        raise ValueError(f"{name} must be non-empty and at most {_MAX_IDENTIFIER_BYTES} bytes")


def validate_command(command: GraphCommand) -> str:
    validate_identifier(command.command_id, "command_id")
    validate_identifier(command.run_id, "run_id")
    validate_identifier(command.invocation_id, "invocation_id")
    validate_identifier(command.owner_incarnation, "owner_incarnation")
    if command.node_id < 1:
        raise ValueError("node_id must be positive")
    if command.action not in _ACTIONS:
        raise ValueError(f"unsupported session action: {command.action!r}")
    if len(command.text.encode()) > _MAX_COMMAND_TEXT_BYTES:
        raise ValueError("command text exceeds 65536 bytes")
    if command.permission_id is not None:
        validate_identifier(command.permission_id, "permission_id")
    return utc_iso(command.expires_at)


def caps_json(values: tuple[str, ...]) -> str:
    return json.dumps(values, separators=(",", ":"))


def decode_strings(value: str, name: str) -> tuple[str, ...]:
    try:
        decoded = cast(object, json.loads(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"stored {name} is invalid JSON") from exc
    items = cast(list[object], decoded)
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"stored {name} must be a string list")
    return tuple(cast(str, item) for item in items)


def capabilities(row: sqlite3.Row | None) -> OwnerCapabilities | None:
    if row is None:
        return None
    actions = cast(tuple[SessionAction, ...], decode_strings(cast(str, row[4]), "owner actions"))
    permission_ids = decode_strings(cast(str, row[5]), "permission IDs")
    return OwnerCapabilities(
        run_id=cast(str, row[0]),
        node_id=cast(int, row[1]),
        invocation_id=cast(str, row[2]),
        owner_incarnation=cast(str, row[3]),
        actions=actions,
        permission_ids=permission_ids,
        published_at=cast(str, row[6]),
    )


def get_capabilities(conn: sqlite3.Connection, run_id: str) -> OwnerCapabilities | None:
    return capabilities(
        fetchone(
            conn,
            "SELECT run_id, node_id, invocation_id, owner_incarnation, actions_json, "  # pyright: ignore[reportImplicitStringConcatenation]
            "permission_ids_json, published_at FROM owner_capabilities WHERE run_id = ?",
            (run_id,),
        )
    )


def command(row: sqlite3.Row | None) -> GraphCommand | None:
    if row is None:
        return None
    return GraphCommand(
        command_id=cast(str, row[1]),
        node_id=cast(int, row[2]),
        run_id=cast(str, row[3]),
        invocation_id=cast(str, row[4]),
        owner_incarnation=cast(str, row[5]),
        action=cast(SessionAction, row[6]),
        text=cast(str, row[7]),
        permission_id=cast(str | None, row[8]),
        expires_at=cast(str, row[9]),
        status=cast(CommandStatus, row[10]),
        admitted_at=cast(str, row[11]),
    )


_COMMAND_SELECT = (
    "SELECT admission_seq, command_id, node_id, run_id, invocation_id, owner_incarnation, "
    "action, text, permission_id, expires_at, status, admitted_at "
    "FROM session_commands "
)


def get_command(conn: sqlite3.Connection, command_id: str) -> GraphCommand | None:
    return command(fetchone(conn, _COMMAND_SELECT + "WHERE command_id = ?", (command_id,)))


def receipt(row: sqlite3.Row | None) -> CommandReceipt | None:
    if row is None:
        return None
    return CommandReceipt(
        command_id=cast(str, row[0]),
        status=cast(CommandStatus, row[1]),
        node_id=cast(int, row[2]),
        run_id=cast(str, row[3]),
        invocation_id=cast(str, row[4]),
        owner_incarnation=cast(str, row[5]),
        action=cast(SessionAction, row[6]),
        text=cast(str, row[7]),
        permission_id=cast(str | None, row[8]),
        expires_at=cast(str, row[9]),
        admitted_at=cast(str, row[10]),
        recorded_at=cast(str, row[11]),
        detail=cast(str | None, row[12]),
    )


_RECEIPT_SELECT = (
    "SELECT c.command_id, r.status, c.node_id, c.run_id, c.invocation_id, "
    "c.owner_incarnation, c.action, c.text, c.permission_id, c.expires_at, "
    "c.admitted_at, r.recorded_at, r.detail "
    "FROM command_receipts r JOIN session_commands c ON c.command_id = r.command_id "
)


def get_receipt(conn: sqlite3.Connection, command_id: str) -> CommandReceipt | None:
    return receipt(
        fetchone(
            conn,
            _RECEIPT_SELECT + "WHERE r.command_id = ? ORDER BY r.receipt_seq DESC LIMIT 1",
            (command_id,),
        )
    )


def receipt_history(conn: sqlite3.Connection, command_id: str) -> tuple[CommandReceipt, ...]:
    rows = fetchall(
        conn,
        _RECEIPT_SELECT + "WHERE r.command_id = ? ORDER BY r.receipt_seq",
        (command_id,),
    )
    return tuple(stored for row in rows if (stored := receipt(row)) is not None)


def same_command(left: GraphCommand, right: GraphCommand, expires_at: str) -> bool:
    return (
        left.node_id == right.node_id
        and left.run_id == right.run_id
        and left.invocation_id == right.invocation_id
        and left.owner_incarnation == right.owner_incarnation
        and left.action == right.action
        and left.text == right.text
        and left.permission_id == right.permission_id
        and left.expires_at == expires_at
    )


def fence_reason(conn: sqlite3.Connection, command_value: GraphCommand) -> str | None:
    run = fetchone(
        conn, "SELECT node_id, status FROM runs WHERE run_id = ?", (command_value.run_id,)
    )
    if run is None or cast(int, run[0]) != command_value.node_id:
        return "run/node fence does not match"
    if cast(str, run[1]) != "running":
        return "run is not running"
    caps = get_capabilities(conn, command_value.run_id)
    if caps is None:
        return "owner capabilities are not published"
    if (
        caps.node_id != command_value.node_id
        or caps.invocation_id != command_value.invocation_id
        or caps.owner_incarnation != command_value.owner_incarnation
    ):
        return "owner incarnation or invocation fence does not match"
    if command_value.action not in caps.actions:
        return f"action {command_value.action!r} is not currently available"
    if command_value.action in {"approve", "deny"}:
        if (
            command_value.permission_id is None
            or command_value.permission_id not in caps.permission_ids
        ):
            return "permission ID does not exactly match a current permission"
    elif command_value.permission_id is not None:
        return "permission ID is valid only for approval commands"
    return None


def record_receipt(  # noqa: PLR0913
    conn: sqlite3.Connection, command_id: str, status: str, now: str, detail: str | None
) -> None:
    _ = conn.execute(
        "INSERT INTO command_receipts (command_id, status, recorded_at, detail) "  # pyright: ignore[reportImplicitStringConcatenation]
        "VALUES (?, ?, ?, ?)",
        (command_id, status, now, detail),
    )


__all__ = [
    "_ACTIONS",
    "_MAX_PENDING_COMMANDS",
    "_TERMINAL",
    "_TRANSITIONS",
    "_RECEIPT_SELECT",
    "capabilities",
    "caps_json",
    "command",
    "decode_strings",
    "fence_reason",
    "get_capabilities",
    "get_command",
    "get_receipt",
    "record_receipt",
    "receipt",
    "same_command",
    "utc_iso",
    "validate_command",
    "validate_identifier",
]
