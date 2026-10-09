from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import cast

import pytest

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
    record_provider_turn,
)
from milknado.domains.graph import MikadoGraph

SESSION_ID = "token=synthetic-provider"
TURN_ID = "sk-SyntheticTurn12345"


def _bound_session(graph: MikadoGraph) -> str:
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session_id = start_coordinator(conn, goal.id, "codex").id
        link_entity(conn, session_id, "provider_session", SESSION_ID)
        bind_provider_session(
            conn, session_id, ProviderBinding("coordinator", session_id, "codex", SESSION_ID)
        )
    return session_id


def test_turn_journal_redacts_identifiers_without_changing_recovery(graph: MikadoGraph) -> None:
    session_id = _bound_session(graph)
    turn = ProviderTurn(ProviderIdentity("codex", SESSION_ID), TURN_ID, "submitted")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        record_provider_turn(conn, session_id, turn)
        recovery = cast(
            tuple[str, str] | None,
            conn.execute(
                "SELECT provider_session_id, turn_id FROM coordinator_turn_events "
                + "WHERE coordinator_id = ?",
                (session_id,),
            ).fetchone(),
        )
        display = [
            event for event in control_history(conn, session_id) if event.kind == "provider_turn"
        ]
    assert recovery == (SESSION_ID, TURN_ID)
    assert len(display) == 1
    assert display[0].entity_id == "token=[REDACTED]"
    assert display[0].tool_name == "[REDACTED]"
    assert SESSION_ID not in repr(display[0])
    assert TURN_ID not in repr(display[0])


def test_display_insert_failure_rolls_back_recovery_turn(graph: MikadoGraph) -> None:
    session_id = _bound_session(graph)
    turn = ProviderTurn(ProviderIdentity("codex", SESSION_ID), TURN_ID, "submitted")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        _ = conn.execute(
            "CREATE TRIGGER reject_provider_turn_display BEFORE INSERT ON coordinator_events "
            + "WHEN NEW.kind = 'provider_turn' BEGIN SELECT RAISE(ABORT, 'display blocked'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="display blocked"):
            record_provider_turn(conn, session_id, turn)
        recovery = cast(
            list[tuple[str, str]],
            conn.execute(
                "SELECT provider_session_id, turn_id FROM coordinator_turn_events "
                + "WHERE coordinator_id = ?",
                (session_id,),
            ).fetchall(),
        )
        display = [
            event for event in control_history(conn, session_id) if event.kind == "provider_turn"
        ]
    assert recovery == []
    assert display == []
