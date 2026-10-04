from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import msgspec

from milknado.domains.common import SessionInput
from milknado.domains.coordinator.journal import append_control_event
from milknado.domains.coordinator.model import ControlEvent, CoordinatorSession
from milknado.domains.coordinator.persistence import link_entity
from milknado.loop.sessions import RuntimeSession, submit_runtime_action


@dataclass(frozen=True)
class CoordinatorActionReceipt:
    command_id: str
    provider_session_id: str
    state: str


@dataclass(frozen=True)
class CoordinatorAction:
    command_id: str
    input: SessionInput


def _create_action_table(conn: sqlite3.Connection) -> None:
    with conn:
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS coordinator_action_receipts (
                command_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES coordinator_sessions(id) ON DELETE CASCADE,
                provider_session_id TEXT NOT NULL,
                action_hash TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def submit_coordinator_action(
    conn: sqlite3.Connection,
    session: CoordinatorSession,
    runtime_session: RuntimeSession,
    command: CoordinatorAction,
) -> CoordinatorActionReceipt:
    if not command.command_id:
        raise ValueError("command identity must not be empty")
    if runtime_session.identity.family != session.provider:
        raise ValueError("provider does not match coordinator session")
    _create_action_table(conn)
    identity = runtime_session.identity.session_id
    fingerprint = hashlib.sha256(msgspec.json.encode(command.input)).hexdigest()
    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO coordinator_action_receipts "
            + "(command_id, session_id, provider_session_id, action_hash, state, created_at) "
            + "VALUES (?, ?, ?, ?, 'unconfirmed', ?)",
            (command.command_id, session.id, identity, fingerprint, datetime.now(UTC).isoformat()),
        )
    row = cast(
        tuple[str, str, str, str] | None,
        conn.execute(
            "SELECT session_id, provider_session_id, action_hash, state "
            + "FROM coordinator_action_receipts WHERE command_id = ?",
            (command.command_id,),
        ).fetchone(),
    )
    if row is None:
        raise RuntimeError("coordinator command has no durable receipt")
    if row[:3] != (session.id, identity, fingerprint):
        raise ValueError("command identity was reused for another action")
    if cursor.rowcount == 0:
        return CoordinatorActionReceipt(command.command_id, identity, row[3])
    result = submit_runtime_action(identity, command.input, runtime_session)
    with conn:
        _ = conn.execute(
            "UPDATE coordinator_action_receipts SET state = ? WHERE command_id = ?",
            (result.state, command.command_id),
        )
    link_entity(conn, session.id, "provider_session", identity)
    _ = append_control_event(
        conn,
        session.id,
        ControlEvent(
            kind="approval" if command.input.action in {"approve", "deny"} else "command",
            entity_kind="provider_session",
            entity_id=identity,
            status=result.state,
        ),
    )
    return CoordinatorActionReceipt(command.command_id, identity, result.state)
