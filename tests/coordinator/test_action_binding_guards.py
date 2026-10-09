from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.common import SessionContext, SessionInput
from milknado.domains.coordinator import ProviderBinding
from milknado.domains.coordinator.commands import CoordinatorAction, submit_coordinator_action
from milknado.domains.coordinator.persistence import bind_provider_session, link_entity
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import MikadoGraph
from milknado.loop.sessions import ProviderSessionIdentity, RuntimeSession, SessionChannel


def test_foreign_family_binding_cannot_authorize_same_session_string(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session = CoordinatorWorkflow(graph, conn).start_goal("Goal", "codex")
        bind_provider_session(
            conn, session.id, ProviderBinding("coordinator", session.id, "claude", "provider")
        )
        link_entity(conn, session.id, "provider_session", "provider")
        channel = SessionChannel()
        channel.start(SessionContext(family="codex", cwd=str(tmp_path)), ("steer",))
        incarnation = channel.capture_incarnation()
        assert incarnation is not None
        runtime = RuntimeSession(
            ProviderSessionIdentity("codex", "provider"), channel, incarnation
        )
        with pytest.raises(ValueError, match="not owned"):
            _ = submit_coordinator_action(
                conn,
                session,
                runtime,
                CoordinatorAction("foreign", SessionInput(action="steer", text="continue")),
            )
        assert conn.execute("SELECT COUNT(*) FROM coordinator_action_receipts").fetchone() == (0,)
        assert channel.drain() == ()
    graph.close()
