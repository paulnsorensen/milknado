"""Durable command admission and receipt transitions for graph-owned sessions."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import cast

from milknado.domains.common.session import SessionAction
from milknado.domains.graph._command_claim import expire_commands_in_transaction
from milknado.domains.graph._command_records import (
    _ACTIONS,
    _MAX_PENDING_COMMANDS,
    _TERMINAL,
    _TRANSITIONS,
    caps_json,
    command,
    fence_reason,
    get_capabilities,
    get_command,
    get_receipt,
    receipt_history,
    record_receipt,
    same_command,
    utc_iso,
    validate_command,
    validate_identifier,
)
from milknado.domains.graph._sqlite_rows import fetchall, fetchone
from milknado.domains.graph.commands import (
    CommandFenceError,
    CommandReceipt,
    CommandStatus,
    GraphCommand,
    OwnerCapabilities,
)


def publish_capabilities(  # noqa: PLR0913
    conn: sqlite3.Connection,
    run_id: str,
    node_id: int,
    invocation_id: str,
    owner_incarnation: str,
    actions: tuple[str, ...],
    permission_ids: tuple[str, ...] = (),
    *,
    published_at: str | None = None,
) -> OwnerCapabilities:
    """Replace the current owner snapshot without deriving it from events."""
    validate_identifier(run_id, "run_id")
    validate_identifier(invocation_id, "invocation_id")
    validate_identifier(owner_incarnation, "owner_incarnation")
    if node_id < 1 or any(action not in _ACTIONS for action in actions):
        raise ValueError("owner capabilities contain an invalid node or action")
    if len(set(actions)) != len(actions) or len(set(permission_ids)) != len(permission_ids):
        raise ValueError("owner capabilities must not contain duplicates")
    for permission_id in permission_ids:
        validate_identifier(permission_id, "permission_id")
    timestamp = utc_iso(published_at)
    with conn:
        run = fetchone(conn, "SELECT node_id, status FROM runs WHERE run_id = ?", (run_id,))
        if run is None or cast(int, run[0]) != node_id:
            raise ValueError("owner capabilities do not match the run node")
        if cast(str, run[1]) != "running":
            raise ValueError("owner capabilities require a running run")
        _ = conn.execute(
            "INSERT INTO owner_capabilities "  # pyright: ignore[reportImplicitStringConcatenation]
            "(run_id, node_id, invocation_id, owner_incarnation, actions_json, "
            "permission_ids_json, published_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET node_id=excluded.node_id, "
            "invocation_id=excluded.invocation_id, owner_incarnation=excluded.owner_incarnation, "
            "actions_json=excluded.actions_json, "
            "permission_ids_json=excluded.permission_ids_json, "
            "published_at=excluded.published_at",
            (
                run_id,
                node_id,
                invocation_id,
                owner_incarnation,
                caps_json(actions),
                caps_json(permission_ids),
                timestamp,
            ),
        )
    return OwnerCapabilities(
        run_id=run_id,
        node_id=node_id,
        invocation_id=invocation_id,
        owner_incarnation=owner_incarnation,
        actions=cast(tuple[SessionAction, ...], actions),
        permission_ids=permission_ids,
        published_at=timestamp,
    )


def admit_command(
    conn: sqlite3.Connection,
    command_value: GraphCommand,
    *,
    now: str | None = None,
    max_pending: int = _MAX_PENDING_COMMANDS,
) -> CommandReceipt:
    """Atomically admit one command or persist its rejection/expiry receipt."""
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    timestamp = utc_iso(now)
    expires_at = validate_command(command_value)
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        existing = get_command(conn, command_value.command_id)
        if existing is not None:
            if not same_command(existing, command_value, expires_at):
                raise ValueError("command_id already names a different command")
            stored_receipt = get_receipt(conn, command_value.command_id)
            if stored_receipt is None:
                raise RuntimeError("stored command has no receipt")
            return stored_receipt
        status: CommandStatus = "queued"
        detail: str | None = None
        if datetime.fromisoformat(expires_at) <= datetime.fromisoformat(timestamp):
            status, detail = "expired", "command expired before admission"
        else:
            detail = fence_reason(conn, command_value)
            if detail is None and _pending_decision(conn, command_value):
                detail = "permission already has a pending decision"
            if detail is not None:
                status = "rejected"
            else:
                queued = fetchone(
                    conn, "SELECT COUNT(*) FROM session_commands WHERE status = 'queued'"
                )
                if queued is None:
                    raise RuntimeError("queued command count returned no row")
                if cast(int, queued[0]) >= max_pending:
                    status, detail = "rejected", "command inbox is full"
        _ = conn.execute(
            """INSERT INTO session_commands
               (command_id, node_id, run_id, invocation_id, owner_incarnation, action, text,
                permission_id, expires_at, status, admitted_at, updated_at, detail)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                command_value.command_id,
                command_value.node_id,
                command_value.run_id,
                command_value.invocation_id,
                command_value.owner_incarnation,
                command_value.action,
                command_value.text,
                command_value.permission_id,
                expires_at,
                status,
                timestamp,
                timestamp,
                detail,
            ),
        )
        record_receipt(conn, command_value.command_id, status, timestamp, detail)
        stored_receipt = get_receipt(conn, command_value.command_id)
    if stored_receipt is None:
        raise RuntimeError("command admission returned no receipt")
    return stored_receipt


def _pending_decision(conn: sqlite3.Connection, value: GraphCommand) -> bool:
    if value.permission_id is None:
        return False
    return (
        fetchone(
            conn,
            """SELECT 1 FROM session_commands
           WHERE run_id = ? AND invocation_id = ? AND owner_incarnation = ?
             AND permission_id = ? AND status IN ('queued', 'submitted')
           LIMIT 1""",
            (value.run_id, value.invocation_id, value.owner_incarnation, value.permission_id),
        )
        is not None
    )


def queued_commands(
    conn: sqlite3.Connection,
    run_id: str | None = None,
    *,
    now: str | None = None,
    limit: int = _MAX_PENDING_COMMANDS,
) -> tuple[GraphCommand, ...]:
    if not 1 <= limit <= _MAX_PENDING_COMMANDS:
        raise ValueError(f"limit must be between 1 and {_MAX_PENDING_COMMANDS}")
    timestamp = utc_iso(now)
    _ = expire_commands(conn, now=timestamp)
    params: tuple[object, ...] = () if run_id is None else (run_id,)
    where = "status = 'queued'" if run_id is None else "status = 'queued' AND run_id = ?"
    rows = fetchall(
        conn,
        "SELECT admission_seq, command_id, node_id, run_id, invocation_id, owner_incarnation, "  # pyright: ignore[reportImplicitStringConcatenation]
        "action, text, permission_id, expires_at, status, admitted_at FROM session_commands WHERE "
        + where
        + " ORDER BY admission_seq LIMIT ?",
        (*params, limit),
    )
    return tuple(command_value for row in rows if (command_value := command(row)) is not None)


def transition_command(  # noqa: PLR0913
    conn: sqlite3.Connection,
    command_id: str,
    status: CommandStatus,
    *,
    node_id: int,
    run_id: str,
    invocation_id: str,
    owner_incarnation: str,
    permission_id: str | None = None,
    now: str | None = None,
    detail: str | None = None,
) -> CommandReceipt:
    """Record a fenced receipt transition without claiming or replaying work."""
    if status not in _TERMINAL | {"submitted"}:
        raise ValueError(f"invalid command transition: {status!r}")
    timestamp = utc_iso(now)
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        command_value = get_command(conn, command_id)
        if command_value is None:
            raise ValueError(f"unknown command: {command_id!r}")
        if (
            command_value.node_id != node_id
            or command_value.run_id != run_id
            or command_value.invocation_id != invocation_id
            or command_value.owner_incarnation != owner_incarnation
            or command_value.permission_id != permission_id
        ):
            raise CommandFenceError("command fence does not match the stored command")
        if command_value.status == status:
            stored_receipt = get_receipt(conn, command_id)
            if stored_receipt is None:
                raise RuntimeError("command has no receipt")
            return stored_receipt
        if command_value.status in _TERMINAL:
            raise ValueError(f"terminal command cannot transition from {command_value.status!r}")
        if datetime.fromisoformat(command_value.expires_at) <= datetime.fromisoformat(timestamp):
            _ = conn.execute(
                "UPDATE session_commands SET status = 'expired', updated_at = ?, detail = ? "  # pyright: ignore[reportImplicitStringConcatenation]
                "WHERE command_id = ? AND status IN ('queued', 'submitted')",
                (timestamp, "command expired before transition", command_id),
            )
            record_receipt(
                conn, command_id, "expired", timestamp, "command expired before transition"
            )
        else:
            reason = fence_reason(conn, command_value)
            if reason is not None:
                raise ValueError(reason)
            if status not in _TRANSITIONS.get(command_value.status, frozenset()):
                raise ValueError(
                    f"invalid command transition {command_value.status!r} -> {status!r}"
                )
            _ = conn.execute(
                "UPDATE session_commands SET status = ?, updated_at = ?, detail = ? "  # pyright: ignore[reportImplicitStringConcatenation]
                "WHERE command_id = ?",
                (status, timestamp, detail, command_id),
            )
            record_receipt(conn, command_id, status, timestamp, detail)
        stored_receipt = get_receipt(conn, command_id)
    if stored_receipt is None:
        raise RuntimeError("command transition returned no receipt")
    return stored_receipt


def expire_commands(
    conn: sqlite3.Connection,
    *,
    now: str | None = None,
) -> tuple[CommandReceipt, ...]:
    timestamp = utc_iso(now)
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        return expire_commands_in_transaction(conn, timestamp)


__all__ = [
    "admit_command",
    "expire_commands",
    "get_capabilities",
    "get_command",
    "get_receipt",
    "publish_capabilities",
    "queued_commands",
    "receipt_history",
    "transition_command",
]
