"""Durable one-writer fences for coordinator provider turns."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

from milknado.domains.common import WorkerOwner
from milknado.domains.coordinator.control_models import StartTurn
from milknado.domains.coordinator.recovery import (
    ProviderIdentity,
    ProviderTurn,
    WorkerTerminationPort,
    record_provider_turn,
)
from milknado.domains.coordinator.turns import TurnLaunch
from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True, slots=True)
class TurnFence:
    command_id: str
    scope_kind: str
    scope_id: str
    supervisor_pid: int
    supervisor_start_token: float


def claim_turn(  # noqa: PLR0913 - launch ownership needs the command, scope, and owner
    conn: sqlite3.Connection,
    session_id: str,
    command: StartTurn,
    launch: TurnLaunch,
    owner: WorkerOwner | None,
) -> None:
    if owner is not None and owner.runtime_run_id != command.command_id:
        raise ValueError("provider turn owner does not match command")
    try:
        with conn:
            _ = conn.execute(
                "INSERT INTO coordinator_turn_launches "
                + "(command_id, coordinator_id, scope_kind, scope_id, state, "
                + "supervisor_pid, supervisor_start_token) "
                + "VALUES (?, ?, ?, ?, 'submitted', ?, ?)",
                (
                    command.command_id,
                    session_id,
                    launch.scope_kind,
                    launch.scope_id,
                    owner.supervisor_pid if owner is not None else None,
                    owner.supervisor_start_token if owner is not None else None,
                ),
            )
    except sqlite3.IntegrityError as error:
        raise ValueError("scope already has an in-flight provider turn") from error
    if launch.identity is not None:
        record_provider_turn(
            conn,
            session_id,
            ProviderTurn(
                ProviderIdentity(launch.provider, launch.identity.session_id),
                command.command_id,
                "submitted",
            ),
        )


def pending_turn_fences(conn: sqlite3.Connection, session_id: str) -> tuple[TurnFence, ...]:
    rows = cast(
        list[tuple[str, str, str, int | None, float | None]],
        conn.execute(
            "SELECT command_id, scope_kind, scope_id, supervisor_pid, "
            + "supervisor_start_token FROM coordinator_turn_launches "
            + "WHERE coordinator_id = ? AND state = 'submitted'",
            (session_id,),
        ).fetchall(),
    )
    return tuple(
        TurnFence(command_id, scope_kind, scope_id, pid, token)
        for command_id, scope_kind, scope_id, pid, token in rows
        if pid is not None and token is not None
    )


def clear_verified_fence(conn: sqlite3.Connection, fence: TurnFence) -> bool:
    with conn:
        _ = conn.execute("BEGIN IMMEDIATE")
        row = cast(
            tuple[str, str, str, int, float] | None,
            conn.execute(
                "SELECT scope_kind, scope_id, state, supervisor_pid, supervisor_start_token "
                + "FROM coordinator_turn_launches WHERE command_id = ?",
                (fence.command_id,),
            ).fetchone(),
        )
        if row is None or tuple(row) != (
            fence.scope_kind,
            fence.scope_id,
            "submitted",
            fence.supervisor_pid,
            fence.supervisor_start_token,
        ):
            return False
        active = cast(
            tuple[int] | None,
            conn.execute(
                "SELECT 1 FROM run_workers WHERE runtime_run_id = ? AND ended_at IS NULL LIMIT 1",
                (fence.command_id,),
            ).fetchone(),
        )
        if active is not None:
            return False
        _ = conn.execute(
            "UPDATE coordinator_turn_launches SET state = 'unknown' WHERE command_id = ?",
            (fence.command_id,),
        )
    return True


def release_unconfirmed_turn(conn: sqlite3.Connection, command_id: str) -> None:
    with conn:
        _ = conn.execute(
            "UPDATE coordinator_turn_launches SET state = 'unknown' "
            + "WHERE command_id = ? AND state = 'submitted'",
            (command_id,),
        )


def cancel_owned_turn(
    conn: sqlite3.Connection,
    session_id: str,
    turn_id: str,
    cancel: Callable[[str], bool] | None,
) -> tuple[Literal["accepted", "unavailable"], object]:
    owned = cast(
        tuple[int] | None,
        conn.execute(
            "SELECT 1 FROM coordinator_turn_launches WHERE command_id = ? "
            + "AND coordinator_id = ? AND state = 'submitted'",
            (turn_id, session_id),
        ).fetchone(),
    )
    if owned is None:
        raise ValueError("active provider turn is not owned by coordinator")
    if cancel is None:
        return "unavailable", "Turn cancellation is not connected."
    if not cancel(turn_id):
        return "unavailable", "Provider turn is not active."
    return "accepted", None


def reconcile_turn_fences(
    conn: sqlite3.Connection,
    graph: MikadoGraph,
    session_id: str,
    workers: WorkerTerminationPort,
) -> None:
    with graph.synchronization_lock:
        pending = pending_turn_fences(conn, session_id)
    for fence in pending:
        if workers.terminated(
            fence.command_id, fence.supervisor_pid, fence.supervisor_start_token
        ):
            with graph.synchronization_lock:
                _ = clear_verified_fence(conn, fence)
