from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

import msgspec

from milknado.domains.common import SessionInput
from milknado.domains.coordinator.journal import append_control_event, operation_identity
from milknado.domains.coordinator.model import ControlEvent, CoordinatorSession
from milknado.domains.graph import TaskAttempt


class ActionSession(Protocol):
    @property
    def family(self) -> str: ...

    @property
    def provider_session_id(self) -> str: ...

    def submit_action(self, action: SessionInput) -> str: ...


@dataclass(frozen=True)
class DispatchHandoff:
    attempt: TaskAttempt
    state: str


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
    try:
        _ = append_control_event(conn, session_id, event)
    except sqlite3.IntegrityError:
        known = cast(
            tuple[int] | None,
            conn.execute(
                "SELECT 1 FROM coordinator_events WHERE session_id = ? AND operation_hash = ?",
                (session_id, operation_identity(event)),
            ).fetchone(),
        )
        if known is None:
            raise


@dataclass(frozen=True)
class CoordinatorActionReceipt:
    command_id: str
    provider_session_id: str
    state: str


@dataclass(frozen=True)
class CoordinatorAction:
    command_id: str
    input: SessionInput


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


def _reserve_action_receipt(
    conn: sqlite3.Connection,
    session: CoordinatorSession,
    identity: str,
    command: CoordinatorAction,
) -> tuple[bool, str]:
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
    return cursor.rowcount == 1, row[3]


def submit_coordinator_action(  # noqa: V103
    conn: sqlite3.Connection,
    session: CoordinatorSession,
    runtime_session: ActionSession,
    command: CoordinatorAction,
) -> CoordinatorActionReceipt:
    if not command.command_id:
        raise ValueError("command identity must not be empty")
    identity = runtime_session.provider_session_id
    linked = cast(
        tuple[int] | None,
        conn.execute(
            "SELECT 1 FROM coordinator_provider_bindings WHERE coordinator_id = ? "
            + "AND provider_family = ? AND provider_session_id = ?",
            (session.id, runtime_session.family, identity),
        ).fetchone(),
    )
    if linked is None:
        raise ValueError("provider session is not owned by coordinator")
    created, state = _reserve_action_receipt(conn, session, identity, command)
    if created:
        state = runtime_session.submit_action(command.input)
        with conn:
            _ = conn.execute(
                "UPDATE coordinator_action_receipts SET state = ? WHERE command_id = ?",
                (state, command.command_id),
            )
    _record_action_event(conn, session.id, command, state)
    return CoordinatorActionReceipt(command.command_id, identity, state)
