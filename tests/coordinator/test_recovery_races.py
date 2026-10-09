from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

import milknado.domains.coordinator.recovery as recovery
from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.coordinator import ProviderBinding
from milknado.domains.coordinator.journal import control_history
from milknado.domains.coordinator.persistence import (
    bind_provider_session,
    link_entity,
    start_coordinator,
)
from milknado.domains.coordinator.recovery import (
    ProviderIdentity,
    ProviderTurn,
    RecoveryRuntime,
    record_provider_turn,
    recover_coordinator,
)
from milknado.domains.graph import ExecutionGroup, MikadoGraph


class ProviderPort:
    def __init__(self) -> None:
        self.calls: list[tuple[ProviderIdentity, Path]] = []

    def recover(self, identity: ProviderIdentity, cwd: Path) -> recovery.RecoveryOutcome:
        self.calls.append((identity, cwd))
        return "unsupported"


class WorktreePort:
    def __init__(self) -> None:
        self.calls: list[ExecutionGroup] = []

    def restore(self, group: ExecutionGroup) -> bool:
        self.calls.append(group)
        return False


def _bound_session(graph: MikadoGraph) -> str:
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session_id = start_coordinator(conn, goal.id, "codex").id
        link_entity(conn, session_id, "provider_session", "provider-1")
        bind_provider_session(
            conn, session_id, ProviderBinding("coordinator", session_id, "codex", "provider-1")
        )
    return session_id


def test_delayed_submission_cannot_regress_confirmed_turn(graph: MikadoGraph) -> None:
    session_id = _bound_session(graph)
    identity = ProviderIdentity("codex", "provider-1")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        for status in ("submitted", "confirmed", "submitted"):
            record_provider_turn(conn, session_id, ProviderTurn(identity, "turn-1", status))
        statuses = [
            event.status
            for event in control_history(conn, session_id)
            if event.kind == "provider_turn"
        ]
    assert statuses == ["submitted", "confirmed"]


def test_concurrent_confirmation_wins_over_stale_recovery_decision(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = _bound_session(graph)
    identity = ProviderIdentity("codex", "provider-1")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        record_provider_turn(conn, session_id, ProviderTurn(identity, "turn-1", "submitted"))
    original = cast(
        Callable[[sqlite3.Connection, str, ProviderTurn], recovery.TurnStatus | None],
        recovery.__dict__["_append_turn_transition"],
    )

    def confirm_before_unknown(
        conn: sqlite3.Connection, coordinator_id: str, turn: ProviderTurn
    ) -> recovery.TurnStatus | None:
        if turn.status == "unknown":
            _ = original(conn, coordinator_id, ProviderTurn(identity, "turn-1", "confirmed"))
        return original(conn, coordinator_id, turn)

    monkeypatch.setattr(recovery, "_append_turn_transition", confirm_before_unknown)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        result = recover_coordinator(
            conn,
            session_id,
            RecoveryRuntime(graph.groups, tmp_path, ProviderPort(), WorktreePort()),
        )
        statuses = [
            event.status
            for event in control_history(conn, session_id)
            if event.kind == "provider_turn"
        ]
    assert statuses == ["submitted", "confirmed"]
    assert result.unknown_turns == ()


def test_deleting_coordinator_cascades_bindings_and_turn_events(graph: MikadoGraph) -> None:
    session_id = _bound_session(graph)
    identity = ProviderIdentity("codex", "provider-1")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        _ = conn.execute("PRAGMA foreign_keys = ON")
        record_provider_turn(conn, session_id, ProviderTurn(identity, "turn-1", "submitted"))
        _ = conn.execute("DELETE FROM coordinator_sessions WHERE id = ?", (session_id,))
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM coordinator_provider_bindings").fetchone() == (
            0,
        )
        assert conn.execute("SELECT COUNT(*) FROM coordinator_turn_events").fetchone() == (0,)
