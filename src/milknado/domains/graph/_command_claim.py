"""Atomic owner claims for graph-owned session commands."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import cast

from milknado.domains.graph._command_records import (
    _MAX_PENDING_COMMANDS,
    command,
    fence_reason,
    get_capabilities,
    get_receipt,
    record_receipt,
    utc_iso,
    validate_identifier,
)
from milknado.domains.graph._sqlite_rows import fetchall
from milknado.domains.graph.commands import CommandReceipt, GraphCommand


def expire_commands_in_transaction(
    conn: sqlite3.Connection,
    timestamp: str,
    *,
    run_id: str | None = None,
) -> tuple[CommandReceipt, ...]:
    receipts: list[CommandReceipt] = []
    where = "status IN ('queued', 'submitted') AND expires_at <= ?"
    params: list[str] = [timestamp]
    if run_id is not None:
        where += " AND run_id = ?"
        params.append(run_id)
    rows = fetchall(
        conn,
        "SELECT command_id FROM session_commands WHERE " + where + " ORDER BY admission_seq",
        tuple(params),
    )
    for row in rows:
        command_id = cast(str, row[0])
        _ = conn.execute(
            "UPDATE session_commands SET status = 'expired', updated_at = ?, detail = "  # pyright: ignore[reportImplicitStringConcatenation]
            "? WHERE command_id = ? AND status IN ('queued', 'submitted')",
            (timestamp, "command expired", command_id),
        )
        record_receipt(conn, command_id, "expired", timestamp, "command expired")
        stored_receipt = get_receipt(conn, command_id)
        if stored_receipt is None:
            raise RuntimeError("expired command has no receipt")
        receipts.append(stored_receipt)
    return tuple(receipts)


def expire_commands(
    conn: sqlite3.Connection,
    *,
    now: str | None = None,
) -> tuple[CommandReceipt, ...]:
    timestamp = utc_iso(now)
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        return expire_commands_in_transaction(conn, timestamp)


@dataclass(frozen=True, slots=True)
class ClaimRequest:
    run_id: str
    owner_incarnation: str


def claim_queued_commands(
    conn: sqlite3.Connection,
    request: ClaimRequest,
    *,
    now: str | None = None,
    limit: int = _MAX_PENDING_COMMANDS,
) -> tuple[GraphCommand, ...]:
    """Atomically claim queued commands for the current owner invocation."""
    if not 1 <= limit <= _MAX_PENDING_COMMANDS:
        raise ValueError(f"limit must be between 1 and {_MAX_PENDING_COMMANDS}")
    validate_identifier(request.run_id, "run_id")
    validate_identifier(request.owner_incarnation, "owner_incarnation")
    timestamp = utc_iso(now)
    claimed: list[GraphCommand] = []
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        capabilities = get_capabilities(conn, request.run_id)
        if capabilities is None or capabilities.owner_incarnation != request.owner_incarnation:
            return ()
        _ = expire_commands_in_transaction(conn, timestamp, run_id=request.run_id)
        rows = fetchall(
            conn,
            "SELECT admission_seq, command_id, node_id, run_id, invocation_id, "  # pyright: ignore[reportImplicitStringConcatenation]
            "owner_incarnation, action, text, permission_id, expires_at, status, admitted_at "
            "FROM session_commands WHERE status = 'queued' AND run_id = ? "
            "AND node_id = ? AND invocation_id = ? AND owner_incarnation = ? "
            "AND expires_at > ? ORDER BY admission_seq LIMIT ?",
            (
                request.run_id,
                capabilities.node_id,
                capabilities.invocation_id,
                request.owner_incarnation,
                timestamp,
                limit,
            ),
        )
        for row in rows:
            command_value = command(row)
            if command_value is None:
                continue
            reason = fence_reason(conn, command_value)
            if reason is not None:
                query = (
                    "UPDATE session_commands SET status = 'rejected', updated_at = ?, detail = ? "
                    + "WHERE command_id = ? AND status = 'queued'"
                )
                result = conn.execute(
                    query,
                    (timestamp, reason, command_value.command_id),
                )
                if result.rowcount == 1:
                    record_receipt(conn, command_value.command_id, "rejected", timestamp, reason)
                continue
            result = conn.execute(
                "UPDATE session_commands SET status = 'submitted', updated_at = ? "  # pyright: ignore[reportImplicitStringConcatenation]
                "WHERE command_id = ? AND status = 'queued' AND invocation_id = ? "
                "AND owner_incarnation = ?",
                (
                    timestamp,
                    command_value.command_id,
                    capabilities.invocation_id,
                    request.owner_incarnation,
                ),
            )
            if result.rowcount != 1:
                continue
            record_receipt(conn, command_value.command_id, "submitted", timestamp, None)
            claimed.append(command_value)
    return tuple(claimed)


__all__ = ["ClaimRequest", "claim_queued_commands", "expire_commands"]
