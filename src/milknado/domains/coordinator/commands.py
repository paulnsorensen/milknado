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
from milknado.domains.graph import TaskAttempt
from milknado.loop.sessions import RuntimeSession, submit_runtime_action


@dataclass(frozen=True)
class DispatchHandoff:
    attempt: TaskAttempt
    state: str


def create_dispatch_table(conn: sqlite3.Connection) -> None:
    with conn:
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS coordinator_dispatches (
                attempt_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES coordinator_sessions(id) ON DELETE CASCADE,
                group_id TEXT NOT NULL,
                node_id INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                state TEXT NOT NULL
            )
        """)


def owned_dispatch_state(conn: sqlite3.Connection, session_id: str, attempt: TaskAttempt) -> str:
    row = cast(
        tuple[str, str, int, str, str] | None,
        conn.execute(
            "SELECT session_id, group_id, node_id, run_id, state "
            + "FROM coordinator_dispatches WHERE attempt_id = ?",
            (attempt.attempt_id,),
        ).fetchone(),
    )
    if row is None or row[:4] != (session_id, attempt.group_id, attempt.node_id, attempt.run_id):
        raise ValueError("attempt belongs to another coordinator")
    return row[4]


def get_dispatch_state(conn: sqlite3.Connection, attempt_id: str) -> str | None:
    row = cast(
        tuple[str] | None,
        conn.execute(
            "SELECT state FROM coordinator_dispatches WHERE attempt_id = ?", (attempt_id,)
        ).fetchone(),
    )
    return row[0] if row is not None else None


def record_control_once(conn: sqlite3.Connection, session_id: str, event: ControlEvent) -> None:
    known = cast(
        tuple[int] | None,
        conn.execute(
            "SELECT 1 FROM coordinator_events WHERE session_id = ? AND kind = ? "
            + "AND entity_kind = ? AND entity_id = ? AND status = ?",
            (session_id, event.kind, event.entity_kind, event.entity_id, event.status),
        ).fetchone(),
    )
    if known is None:
        _ = append_control_event(conn, session_id, event)


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


def _record_action_event(
    conn: sqlite3.Connection, session_id: str, command: CoordinatorAction, state: str
) -> None:
    record_control_once(
        conn,
        session_id,
        ControlEvent(
            kind="approval" if command.input.action in {"approve", "deny"} else "command",
            entity_kind="coordinator_command",
            entity_id=command.command_id,
            status=state,
        ),
    )


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
    linked = cast(
        tuple[int] | None,
        conn.execute(
            "SELECT 1 FROM coordinator_links WHERE session_id = ? "
            + "AND kind = 'provider_session' AND entity_id = ?",
            (session.id, identity),
        ).fetchone(),
    )
    if linked is None:
        raise ValueError("provider session is not owned by coordinator")
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
        _record_action_event(conn, session.id, command, row[3])
        return CoordinatorActionReceipt(command.command_id, identity, row[3])
    result = submit_runtime_action(identity, command.input, runtime_session)
    with conn:
        _ = conn.execute(
            "UPDATE coordinator_action_receipts SET state = ? WHERE command_id = ?",
            (result.state, command.command_id),
        )
    _record_action_event(conn, session.id, command, result.state)
    return CoordinatorActionReceipt(command.command_id, identity, result.state)
