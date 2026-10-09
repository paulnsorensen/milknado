from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import msgspec

from milknado.domains.common import SessionEvent
from milknado.domains.coordinator.commands import owned_dispatch_state
from milknado.domains.coordinator.control_models import StartTurn
from milknado.domains.coordinator.control_services import (
    TurnIdentity,
    TurnRuntimeHooks,
    TurnRuntimeResult,
)
from milknado.domains.coordinator.journal import (
    append_control_event,
    append_stream_control_event,
)
from milknado.domains.coordinator.model import (
    ControlEvent,
    CoordinatorSession,
    ProviderIdentity,
    RecoveryReceipt,
)
from milknado.domains.coordinator.persistence import provider_bindings_for_session
from milknado.domains.coordinator.recovery import ProviderTurn, record_provider_turn
from milknado.domains.coordinator.recovery_receipts import record_recovery_receipt
from milknado.domains.coordinator.turn_context import TurnContext
from milknado.domains.graph import (
    ExecutionGroup,
    MikadoGraph,
    TaskAttempt,
    bind_execution_group_provider,
)


@dataclass(frozen=True, slots=True)
class TurnLaunch:
    provider: str
    group: ExecutionGroup | None
    identity: TurnIdentity | None
    scope_kind: str
    scope_id: str
    attempt: TaskAttempt | None


def _owned_group(
    conn: sqlite3.Connection, graph: MikadoGraph, session: CoordinatorSession, command: StartTurn
) -> ExecutionGroup:
    assert command.group_id is not None
    group = graph.groups.get(command.group_id)
    linked = cast(
        tuple[int] | None,
        conn.execute(
            "SELECT 1 FROM coordinator_links WHERE session_id = ? "
            + "AND kind = 'execution_group' AND entity_id = ?",
            (session.id, command.group_id),
        ).fetchone(),
    )
    attempt = graph.groups.active_attempt(command.group_id)
    admission = graph.goal_admission(attempt.node_id) if attempt is not None else None
    if (
        group is None
        or linked is None
        or attempt is None
        or (attempt.node_id, attempt.run_id, attempt.attempt_id)
        != (command.node_id, command.run_id, command.attempt_id)
        or admission is None
        or not admission.allowed
        or admission.goal_id != session.goal_id
    ):
        raise ValueError("turn does not own the active execution group writer")
    if owned_dispatch_state(conn, session.id, attempt) != "launched":
        raise ValueError("execution group writer is not launched")
    return group


def _resolve_provider_binding(
    conn: sqlite3.Connection,
    session: CoordinatorSession,
    command: StartTurn,
    group: ExecutionGroup | None,
) -> tuple[str, TurnIdentity | None, str, str]:
    scope_kind = "execution_group" if group else "coordinator"
    scope_id = group.id if group else session.id
    binding = next(
        (
            item
            for item in provider_bindings_for_session(conn, session.id)
            if (item.scope_kind, item.scope_id) == (scope_kind, scope_id)
        ),
        None,
    )
    provider = binding.family if binding else command.provider or session.provider
    if group is None and provider != session.provider:
        raise ValueError("coordinator turn provider conflicts with session provider")
    if command.provider is not None and command.provider != provider:
        raise ValueError("turn provider conflicts with existing binding")
    if provider not in {"claude", "codex"}:
        raise ValueError("unsupported turn provider")
    if group and group.provider_session_id != (binding.provider_session_id if binding else None):
        raise ValueError("execution group provider binding mismatch")
    identity = TurnIdentity(provider, binding.provider_session_id) if binding else None
    return provider, identity, scope_kind, scope_id


def prepare_turn(
    conn: sqlite3.Connection,
    graph: MikadoGraph,
    session: CoordinatorSession,
    command: StartTurn,
) -> TurnLaunch:
    if not command.prompt.strip():
        raise ValueError("turn prompt must not be empty")
    if command.group_id == "":
        raise ValueError("group turn requires a non-empty group ID")
    fields = (command.group_id, command.node_id, command.run_id, command.attempt_id)
    if any(value is not None for value in fields) and not all(
        value is not None for value in fields
    ):
        raise ValueError("group turn requires a complete writer attempt")
    group = _owned_group(conn, graph, session, command) if command.group_id is not None else None
    provider, identity, scope_kind, scope_id = _resolve_provider_binding(
        conn, session, command, group
    )
    attempt = (
        TaskAttempt(
            group.id,
            cast(int, command.node_id),
            cast(str, command.run_id),
            cast(str, command.attempt_id),
        )
        if group is not None
        else None
    )
    return TurnLaunch(provider, group, identity, scope_kind, scope_id, attempt)


def bind_confirmed_identity(
    conn: sqlite3.Connection, session_id: str, launch: TurnLaunch, provider_id: str
) -> None:
    if not provider_id:
        raise ValueError("provider did not confirm session identity")
    with conn:
        _ = conn.execute("BEGIN IMMEDIATE")
        existing = cast(
            tuple[str, str] | None,
            conn.execute(
                "SELECT provider_family, provider_session_id FROM coordinator_provider_bindings "
                + "WHERE coordinator_id = ? AND scope_kind = ? AND scope_id = ?",
                (session_id, launch.scope_kind, launch.scope_id),
            ).fetchone(),
        )
        if existing is not None and tuple(existing) != (launch.provider, provider_id):
            raise ValueError("provider identity conflicts with existing binding")
        if launch.group is not None:
            bind_execution_group_provider(conn, launch.group.id, provider_id)
        _ = conn.execute(
            "INSERT OR IGNORE INTO coordinator_provider_bindings "
            + "(coordinator_id, scope_kind, scope_id, provider_family, provider_session_id) "
            + "VALUES (?, ?, ?, ?, ?)",
            (session_id, launch.scope_kind, launch.scope_id, launch.provider, provider_id),
        )
        bound = cast(
            tuple[str, str, str] | None,
            conn.execute(
                "SELECT coordinator_id, scope_kind, scope_id FROM coordinator_provider_bindings "
                + "WHERE provider_family = ? AND provider_session_id = ?",
                (launch.provider, provider_id),
            ).fetchone(),
        )
        if bound is None or tuple(bound) != (session_id, launch.scope_kind, launch.scope_id):
            raise ValueError("provider identity belongs to another scope")
        _ = conn.execute(
            "INSERT OR IGNORE INTO coordinator_links (session_id, kind, entity_id) "
            + "VALUES (?, 'provider_session', ?)",
            (session_id, provider_id),
        )


class TurnResponse(msgspec.Struct, frozen=True):
    provider_session_id: str
    turn_id: str


def finish_turn(
    context: TurnContext, launch: TurnLaunch, result: TurnRuntimeResult, root: Path
) -> tuple[Literal["accepted", "unavailable"], object]:
    conn, session_id, command_id = context.conn, context.session_id, context.command_id
    run = result.run
    if run is None or not run.session_id:
        return "unavailable", "Provider did not confirm a session identity."
    if launch.identity is not None and run.session_id != launch.identity.session_id:
        return "unavailable", "Provider resumed a different session."
    bind_confirmed_identity(conn, session_id, launch, run.session_id)
    identity = ProviderIdentity(launch.provider, run.session_id)
    record_provider_turn(conn, session_id, ProviderTurn(identity, command_id, "submitted"))
    if not run.terminal_confirmed or (
        launch.identity is not None and not result.recovery_turn_confirmed
    ):
        return "unavailable", "Provider turn has no confirmed terminal result."
    record_provider_turn(conn, session_id, ProviderTurn(identity, command_id, "confirmed"))
    if launch.identity is not None and result.recovery_turn_confirmed is not None:
        _ = record_recovery_receipt(
            conn,
            session_id,
            RecoveryReceipt(
                launch.scope_kind,
                launch.scope_id,
                identity,
                Path(launch.group.worktree_path) if launch.group else root,
                "resumed",
            ),
        )
    with conn:
        _ = conn.execute(
            "UPDATE coordinator_turn_launches SET state = 'confirmed' WHERE command_id = ?",
            (command_id,),
        )
    return "accepted", TurnResponse(run.session_id, command_id)


def confirm_turn_identity(context: TurnContext, launch: TurnLaunch, provider_id: str) -> None:
    conn, session_id, command_id = context.conn, context.session_id, context.command_id
    if launch.identity is not None and provider_id != launch.identity.session_id:
        raise ValueError("provider resumed a different session")
    bind_confirmed_identity(conn, session_id, launch, provider_id)
    record_provider_turn(
        conn,
        session_id,
        ProviderTurn(ProviderIdentity(launch.provider, provider_id), command_id, "submitted"),
    )


def record_turn_event(context: TurnContext, event: SessionEvent, provider_id: str | None) -> None:
    conn, session_id, command_id = context.conn, context.session_id, context.command_id
    if event.kind == "permission" and not provider_id:
        raise ValueError("permission event has no confirmed provider identity")
    control = ControlEvent(
        kind=event.kind,
        text=event.text,
        entity_kind="permission" if event.kind == "permission" else "provider_turn",
        entity_id=event.event_id if event.kind == "permission" else command_id,
        tool_name=event.text if event.kind == "tool" else "",
        status=event.state,
        turn_id=command_id,
        provider_session_id=provider_id or "",
    )
    if event.kind in {"assistant", "error"} and event.event_id:
        _ = append_stream_control_event(conn, session_id, control, event.event_id)
    else:
        _ = append_control_event(conn, session_id, control)


def make_turn_hooks(
    context: TurnContext, graph: MikadoGraph, launch: TurnLaunch
) -> TurnRuntimeHooks:
    provider_id = launch.identity.session_id if launch.identity else None

    def confirm(confirmed_id: str) -> None:
        nonlocal provider_id
        with graph.synchronization_lock:
            confirm_turn_identity(context, launch, confirmed_id)
            provider_id = confirmed_id

    def publish(event: SessionEvent) -> None:
        with graph.synchronization_lock:
            record_turn_event(context, event, provider_id)

    return TurnRuntimeHooks(context.command_id, confirm, publish)
