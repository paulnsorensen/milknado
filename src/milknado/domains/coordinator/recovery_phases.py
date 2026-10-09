"""Snapshot recovery ownership before external provider and worktree probes."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from milknado.domains.coordinator.model import (
    CoordinatorSession,
    ProviderBinding,
    ProviderIdentity,
    RecoveryOutcome,
    RecoveryReceipt,
)
from milknado.domains.coordinator.persistence import (
    get_coordinator,
    links_for_session,
    provider_bindings_for_session,
)
from milknado.domains.graph import ExecutionGroup, ExecutionGroupStore


class ProviderRecoveryPort(Protocol):
    def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome: ...


class WorktreeRecoveryPort(Protocol):
    def restore(self, group: ExecutionGroup) -> bool: ...


class WorkerTerminationPort(Protocol):
    def terminated(
        self, turn_id: str, supervisor_pid: int, supervisor_start_token: float
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class RecoveryRuntime:
    groups: ExecutionGroupStore
    root: Path
    provider: ProviderRecoveryPort
    worktrees: WorktreeRecoveryPort
    workers: WorkerTerminationPort | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedSession:
    binding: ProviderBinding
    identity: ProviderIdentity
    path: Path
    group: ExecutionGroup | None = None


@dataclass(frozen=True, slots=True)
class RecoverySnapshot:
    session: CoordinatorSession
    links: frozenset[tuple[str, str]]
    bindings: tuple[ProviderBinding, ...]
    resolved: tuple[_ResolvedSession, ...]
    groups: tuple[tuple[str, ExecutionGroup], ...]
    turn_events: tuple[tuple[int, str, str, str, str], ...]


def _turn_events(
    conn: sqlite3.Connection, session_id: str
) -> tuple[tuple[int, str, str, str, str], ...]:
    rows = cast(
        list[tuple[int, str, str, str, str]],
        conn.execute(
            "SELECT seq, provider_family, provider_session_id, turn_id, status "
            + "FROM coordinator_turn_events WHERE coordinator_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall(),
    )
    return tuple((row[0], row[1], row[2], row[3], row[4]) for row in rows)


def _resolve_sessions(
    session: CoordinatorSession,
    bindings: tuple[ProviderBinding, ...],
    links: frozenset[tuple[str, str]],
    runtime: RecoveryRuntime,
) -> tuple[_ResolvedSession, ...]:
    expected = {link for link in links if link[0] == "provider_session"}
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
    if expected:
        raise ValueError("coordinator has unbound provider session links")
    return tuple(resolved)


def snapshot_recovery(
    conn: sqlite3.Connection, session_id: str, runtime: RecoveryRuntime
) -> RecoverySnapshot:
    session = get_coordinator(conn, session_id)
    if session is None:
        raise ValueError("coordinator session does not exist")
    if not runtime.root.is_absolute():
        raise ValueError("project root must be absolute")
    links = frozenset((link.kind, link.entity_id) for link in links_for_session(conn, session_id))
    bindings = provider_bindings_for_session(conn, session_id)
    resolved = _resolve_sessions(session, bindings, links, runtime)
    groups: list[tuple[str, ExecutionGroup]] = []
    for kind, group_id in sorted(links):
        if kind != "execution_group":
            continue
        group = runtime.groups.get(group_id)
        if group is None:
            raise ValueError("linked execution group does not exist")
        groups.append((group_id, group))
    return RecoverySnapshot(
        session, links, bindings, resolved, tuple(groups), _turn_events(conn, session_id)
    )


def probe_recovery(
    snapshot: RecoverySnapshot, runtime: RecoveryRuntime
) -> tuple[RecoveryReceipt, ...]:
    restored = {group_id: runtime.worktrees.restore(group) for group_id, group in snapshot.groups}
    receipts = tuple(
        RecoveryReceipt(
            item.binding.scope_kind,
            item.binding.scope_id,
            item.identity,
            item.path,
            runtime.provider.recover(item.identity, item.path)
            if item.group is None or restored[item.binding.scope_id]
            else "unavailable",
        )
        for item in snapshot.resolved
    )
    bound_groups = {item.binding.scope_id for item in snapshot.resolved if item.group is not None}
    return receipts + tuple(
        RecoveryReceipt(
            "execution_group", group_id, None, Path(group.worktree_path), "unavailable"
        )
        for group_id, group in snapshot.groups
        if group_id not in bound_groups
    )


def validate_recovery(
    conn: sqlite3.Connection, snapshot: RecoverySnapshot, runtime: RecoveryRuntime
) -> None:
    session_id = snapshot.session.id
    links = frozenset((link.kind, link.entity_id) for link in links_for_session(conn, session_id))
    if (
        get_coordinator(conn, session_id) != snapshot.session
        or links != snapshot.links
        or provider_bindings_for_session(conn, session_id) != snapshot.bindings
        or _turn_events(conn, session_id) != snapshot.turn_events
        or any(runtime.groups.get(group_id) != group for group_id, group in snapshot.groups)
    ):
        raise ValueError("coordinator recovery inputs changed during probe")
