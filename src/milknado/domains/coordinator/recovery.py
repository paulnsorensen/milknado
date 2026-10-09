from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from milknado.domains.coordinator.journal import redact_control_text
from milknado.domains.coordinator.model import CoordinatorSession
from milknado.domains.coordinator.model import (
    ProviderIdentity as ProviderIdentity,
)
from milknado.domains.coordinator.model import RecoveryOutcome as RecoveryOutcome
from milknado.domains.coordinator.model import (
    RecoveryReceipt as RecoveryReceipt,
)
from milknado.domains.coordinator.recovery_phases import (
    ProviderRecoveryPort as ProviderRecoveryPort,
)
from milknado.domains.coordinator.recovery_phases import (
    RecoveryRuntime as RecoveryRuntime,
)
from milknado.domains.coordinator.recovery_phases import (
    WorkerTerminationPort as WorkerTerminationPort,
)
from milknado.domains.coordinator.recovery_phases import (
    WorktreeRecoveryPort as WorktreeRecoveryPort,
)
from milknado.domains.coordinator.recovery_phases import (
    probe_recovery,
    snapshot_recovery,
    validate_recovery,
)
from milknado.domains.coordinator.recovery_receipts import record_recovery_receipt

TurnStatus = Literal["submitted", "confirmed", "unknown"]


@dataclass(frozen=True, slots=True)
class ProviderTurn:
    identity: ProviderIdentity
    turn_id: str
    status: TurnStatus


@dataclass(frozen=True, slots=True)
class UnknownTurn:
    identity: ProviderIdentity
    turn_id: str


@dataclass(frozen=True, slots=True)
class CoordinatorRecovery:
    session: CoordinatorSession
    receipts: tuple[RecoveryReceipt, ...]
    unknown_turns: tuple[UnknownTurn, ...]  # noqa: V107


def _latest_turn_status(
    conn: sqlite3.Connection, coordinator_id: str, turn: ProviderTurn
) -> str | None:
    row = cast(
        tuple[str] | None,
        conn.execute(
            "SELECT status FROM coordinator_turn_events WHERE coordinator_id = ? "
            + "AND provider_family = ? AND provider_session_id = ? AND turn_id = ? "
            + "ORDER BY seq DESC LIMIT 1",
            (coordinator_id, turn.identity.family, turn.identity.session_id, turn.turn_id),
        ).fetchone(),
    )
    return row[0] if row is not None else None


def _write_turn_event(conn: sqlite3.Connection, coordinator_id: str, turn: ProviderTurn) -> None:
    timestamp = datetime.now(UTC).isoformat()
    values = (
        coordinator_id,
        turn.identity.family,
        turn.identity.session_id,
        turn.turn_id,
        turn.status,
        timestamp,
    )
    _ = conn.execute(
        "INSERT INTO coordinator_turn_events "
        + "(coordinator_id, provider_family, provider_session_id, "
        + "turn_id, status, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
        values,
    )
    _ = conn.execute(
        "INSERT INTO coordinator_events "
        + "(session_id, kind, text, entity_kind, entity_id, tool_name, status, created_at) "
        + "VALUES (?, 'provider_turn', 'provider turn transition', ?, ?, ?, ?, ?)",
        (
            coordinator_id,
            redact_control_text(turn.identity.family),
            redact_control_text(turn.identity.session_id),
            redact_control_text(turn.turn_id),
            redact_control_text(turn.status),
            timestamp,
        ),
    )


def _transition_allowed(current: str | None, target: TurnStatus) -> bool:
    if current == "confirmed" or current == target:
        return False
    if target == "submitted":
        return current is None
    return target == "confirmed" or current == "submitted"


def _append_turn_transition(
    conn: sqlite3.Connection, coordinator_id: str, turn: ProviderTurn
) -> TurnStatus | None:
    if not turn.turn_id:
        raise ValueError("provider turn identity must not be empty")
    with conn:
        _ = conn.execute("BEGIN IMMEDIATE")
        bound = cast(
            tuple[int] | None,
            conn.execute(
                "SELECT 1 FROM coordinator_provider_bindings WHERE coordinator_id = ? "
                + "AND provider_family = ? AND provider_session_id = ?",
                (coordinator_id, turn.identity.family, turn.identity.session_id),
            ).fetchone(),
        )
        if bound is None:
            raise ValueError("provider turn has no coordinator binding")
        current = _latest_turn_status(conn, coordinator_id, turn)
        if _transition_allowed(current, turn.status):
            _write_turn_event(conn, coordinator_id, turn)
            return turn.status
        return cast(TurnStatus | None, current)


def record_provider_turn(  # noqa: V103
    conn: sqlite3.Connection, coordinator_id: str, turn: ProviderTurn
) -> None:
    if turn.status not in {"submitted", "confirmed"}:
        raise ValueError("only provider evidence can confirm a turn")
    _ = _append_turn_transition(conn, coordinator_id, turn)


def _mark_unknown_turns(conn: sqlite3.Connection, coordinator_id: str) -> tuple[UnknownTurn, ...]:
    rows = cast(
        list[tuple[str, str, str, str]],
        conn.execute(
            "SELECT provider_family, provider_session_id, turn_id, status "
            + "FROM coordinator_turn_events WHERE coordinator_id = ? ORDER BY seq",
            (coordinator_id,),
        ).fetchall(),
    )
    states: dict[tuple[str, str, str], str] = {}
    for family, provider_id, turn_id, status in rows:
        states[(family, provider_id, turn_id)] = status
    unknown: list[UnknownTurn] = []
    for (family, provider_id, turn_id), status in states.items():
        if status not in {"submitted", "unknown"}:
            continue
        identity = ProviderIdentity(family, provider_id)
        persisted = _append_turn_transition(
            conn, coordinator_id, ProviderTurn(identity, turn_id, "unknown")
        )
        if persisted == "unknown":
            unknown.append(UnknownTurn(identity, turn_id))
    return tuple(unknown)


def recover_coordinator(  # noqa: V103
    conn: sqlite3.Connection,
    session_id: str,
    runtime: RecoveryRuntime,
    *,
    synchronization_lock: AbstractContextManager[object] | None = None,
) -> CoordinatorRecovery:
    """Probe external recovery outside the lock, then fence its evidence."""
    lock = synchronization_lock or nullcontext()
    with lock:
        snapshot = snapshot_recovery(conn, session_id, runtime)
    candidates = probe_recovery(snapshot, runtime)
    with lock:
        validate_recovery(conn, snapshot, runtime)
        receipts = tuple(
            record_recovery_receipt(conn, session_id, receipt) for receipt in candidates
        )
        unknown = _mark_unknown_turns(conn, session_id)
    return CoordinatorRecovery(snapshot.session, receipts, unknown)
