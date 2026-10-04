from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.coordinator import (
    ControlEvent,
    append_control_event,
    control_history,
    get_coordinator,
    link_entity,
    links_for_session,
    start_coordinator,
)
from milknado.domains.graph._mutations import delete_subtree
from milknado.domains.graph._persistence import create_tables, migrate


def _database(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ = conn.execute("PRAGMA foreign_keys = ON")
    create_tables(conn)
    migrate(conn)
    _ = conn.execute(
        "INSERT INTO nodes (id, description, kind, created_at) VALUES (1, 'goal', 'goal', 'now')"
    )
    _ = conn.execute(
        "INSERT INTO nodes (id, description, kind, parent_id, created_at) "
        + "VALUES (2, 'child', 'goal', 1, 'now'), "
        + "(3, 'nested', 'goal', 2, 'now')"
    )
    _ = conn.execute(
        "INSERT INTO nodes (id, description, kind, created_at) "
        + "VALUES (4, 'roadmap', 'roadmap', 'now')"
    )
    _ = conn.execute(
        "INSERT INTO nodes (id, description, kind, parent_id, created_at) "
        + "VALUES (5, 'roadmap goal', 'goal', 4, 'now')"
    )
    conn.commit()
    return conn


def test_session_identity_and_relationships_survive_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "graph.db")
    with closing(_database(path)) as conn:
        session = start_coordinator(conn, 1, "claude")
        assert start_coordinator(conn, 1, "claude") == session
        link_entity(conn, session.id, "planning_decision", "plan-1")
        link_entity(conn, session.id, "graph_revision", "3")
        link_entity(conn, session.id, "approval", "approval-1")
        link_entity(conn, session.id, "run", "run-1")
        link_entity(conn, session.id, "provider_session", "provider-1")
        link_entity(conn, session.id, "execution_group", "group-1")
        link_entity(conn, session.id, "recovery", "receipt-1")
    with closing(sqlite3.connect(path)) as conn:
        assert get_coordinator(conn, session.id) == session
        assert [(link.kind, link.entity_id) for link in links_for_session(conn, session.id)] == [
            ("planning_decision", "plan-1"),
            ("graph_revision", "3"),
            ("approval", "approval-1"),
            ("run", "run-1"),
            ("provider_session", "provider-1"),
            ("execution_group", "group-1"),
            ("recovery", "receipt-1"),
        ]


def test_only_top_level_goals_start_sessions(tmp_path: Path) -> None:
    with closing(_database(str(tmp_path / "graph.db"))) as conn:
        assert start_coordinator(conn, 2, "claude").goal_id == 2
        assert start_coordinator(conn, 5, "claude").goal_id == 5
        with pytest.raises(ValueError, match="top-level goal"):
            _ = start_coordinator(conn, 3, "claude")
        with pytest.raises(ValueError, match="top-level goal"):
            _ = start_coordinator(conn, 999, "claude")
        with pytest.raises(ValueError, match="provider"):
            _ = start_coordinator(conn, 1, "")


def test_session_rejects_provider_changes_and_invalid_links(tmp_path: Path) -> None:
    with closing(_database(str(tmp_path / "graph.db"))) as conn:
        session = start_coordinator(conn, 1, "claude")
        with pytest.raises(ValueError, match="different provider"):
            _ = start_coordinator(conn, 1, "codex")
        with pytest.raises(ValueError, match="link kind"):
            link_entity(conn, session.id, "unknown", "item-1")
        with pytest.raises(ValueError, match="entity_id"):
            link_entity(conn, session.id, "run", "")


def test_deleting_goal_cascades_coordinator_records(tmp_path: Path) -> None:
    with closing(_database(str(tmp_path / "graph.db"))) as conn:
        _ = conn.execute("INSERT INTO edges (parent_id, child_id) VALUES (1, 2), (2, 3)")
        conn.commit()
        session = start_coordinator(conn, 1, "claude")
        link_entity(conn, session.id, "approval", "approval-1")
        _ = append_control_event(conn, session.id, ControlEvent(kind="command", text="run"))
        assert delete_subtree(conn, 1, True) == 3
        assert get_coordinator(conn, session.id) is None
        assert links_for_session(conn, session.id) == ()
        assert control_history(conn, session.id) == ()


def test_session_starts_with_default_tuple_rows(tmp_path: Path) -> None:
    path = str(tmp_path / "graph.db")
    _database(path).close()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.row_factory is None
        session = start_coordinator(conn, 1, "claude")
        assert conn.row_factory is None
        assert session.goal_id == 1
        assert get_coordinator(conn, session.id) == session
