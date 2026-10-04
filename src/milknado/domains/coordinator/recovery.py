from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from milknado.domains.coordinator.journal import append_control_event
from milknado.domains.coordinator.model import (
    ControlEvent,
    CoordinatorSession,
    ProviderBinding,
)
from milknado.domains.coordinator.persistence import (
    get_coordinator,
    link_entity,
    links_for_session,
    provider_bindings_for_session,
)
from milknado.domains.graph import ExecutionGroup, ExecutionGroupStore

RecoveryOutcome = Literal["reattached", "resumed", "unknown_turn", "unavailable", "unsupported"]
TurnStatus = Literal["submitted", "confirmed", "unknown"]
_OUTCOMES = frozenset({"reattached", "resumed", "unknown_turn", "unavailable", "unsupported"})


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    family: str
    session_id: str

    def __post_init__(self) -> None:
        if self.family not in {"claude", "codex"} or not self.session_id:
            raise ValueError("recovery requires a supported provider session identity")


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
class RecoveryReceipt:
    entity_kind: str
    entity_id: str
    identity: ProviderIdentity
    worktree_path: Path
    outcome: RecoveryOutcome


@dataclass(frozen=True, slots=True)
class CoordinatorRecovery:
    session: CoordinatorSession
    receipts: tuple[RecoveryReceipt, ...]
    unknown_turns: tuple[UnknownTurn, ...]  # noqa: V107


class ProviderRecoveryPort(Protocol):
    def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome: ...


class WorktreeRecoveryPort(Protocol):
    def restore(self, group: ExecutionGroup) -> bool: ...


@dataclass(frozen=True, slots=True)
class RecoveryRuntime:
    groups: ExecutionGroupStore
    root: Path
    provider: ProviderRecoveryPort
    worktrees: WorktreeRecoveryPort


@dataclass(frozen=True, slots=True)
class _ResolvedSession:
    binding: ProviderBinding
    identity: ProviderIdentity
    path: Path
    group: ExecutionGroup | None = None


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
        values,
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


def _resolve_sessions(
    session: CoordinatorSession,
    bindings: tuple[ProviderBinding, ...],
    links: set[tuple[str, str]],
    runtime: RecoveryRuntime,
) -> tuple[_ResolvedSession, ...]:
    expected = {
        (kind, entity_id)
        for kind, entity_id in links
        if kind in {"provider_session", "execution_group"}
    }
    resolved: list[_ResolvedSession] = []
    for binding in bindings:
        identity = ProviderIdentity(binding.family, binding.provider_session_id)
        if ("provider_session", identity.session_id) not in links:
            raise ValueError("provider session binding is not linked to coordinator")
        if binding.scope_kind == "coordinator":
            if binding.scope_id != session.id or identity.family != session.provider:
                raise ValueError("coordinator provider binding identity mismatch")
            resolved.append(_ResolvedSession(binding, identity, runtime.root))
        else:
            if ("execution_group", binding.scope_id) not in links:
                raise ValueError("execution group binding is not linked to coordinator")
            group = runtime.groups.get(binding.scope_id)
            if group is None or group.provider_session_id != identity.session_id:
                raise ValueError("execution group provider identity mismatch")
            path = Path(group.worktree_path)
            if not path.is_absolute() or not group.branch_name:
                raise ValueError("execution group worktree identity is invalid")
            resolved.append(_ResolvedSession(binding, identity, path, group))
        expected.discard(("provider_session", identity.session_id))
        expected.discard(("execution_group", binding.scope_id))
    if expected:
        raise ValueError("coordinator has unbound provider or execution group links")
    return tuple(resolved)


def _record_receipt(
    conn: sqlite3.Connection, session_id: str, receipt: RecoveryReceipt
) -> RecoveryReceipt:
    if receipt.outcome not in _OUTCOMES:
        raise ValueError(f"invalid recovery outcome: {receipt.outcome}")
    seq = append_control_event(
        conn,
        session_id,
        ControlEvent(
            kind="recovery",
            text="provider recovery result",
            entity_kind=receipt.entity_kind,
            entity_id=receipt.entity_id,
            status=receipt.outcome,
        ),
    )
    link_entity(conn, session_id, "recovery", str(seq))
    return receipt


def recover_coordinator(  # noqa: V103
    conn: sqlite3.Connection, session_id: str, runtime: RecoveryRuntime
) -> CoordinatorRecovery:
    """Resolve all identities before restoring worktrees or provider sessions."""
    session = get_coordinator(conn, session_id)
    if session is None:
        raise ValueError("coordinator session does not exist")
    if not runtime.root.is_absolute():
        raise ValueError("project root must be absolute")
    links = {(link.kind, link.entity_id) for link in links_for_session(conn, session_id)}
    bindings = provider_bindings_for_session(conn, session_id)
    resolved = _resolve_sessions(session, bindings, links, runtime)
    restored = {
        item.binding.scope_id: runtime.worktrees.restore(item.group)
        for item in resolved
        if item.group is not None
    }
    receipts = tuple(
        _record_receipt(
            conn,
            session_id,
            RecoveryReceipt(
                item.binding.scope_kind,
                item.binding.scope_id,
                item.identity,
                item.path,
                runtime.provider.recover(item.identity, item.path)
                if item.group is None or restored[item.binding.scope_id]
                else "unavailable",
            ),
        )
        for item in resolved
    )
    return CoordinatorRecovery(session, receipts, _mark_unknown_turns(conn, session_id))
