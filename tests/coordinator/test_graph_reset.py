from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import StartGoal
from milknado.domains.graph import GroupWorkspace, MikadoGraph
from milknado.domains.graph._persistence import SCHEMA_VERSION

COORDINATOR_TABLES = (
    "coordinator_sessions",
    "coordinator_links",
    "coordinator_events",
    "coordinator_turn_launches",
    "coordinator_web_receipts",
    "coordinator_dispatches",
    "coordinator_action_receipts",
    "coordinator_plans",
    "coordinator_plan_proposals",
    "coordinator_provider_bindings",
    "coordinator_turn_events",
)
GROUP_TABLES = ("execution_group_tasks", "graph_alternatives", "execution_groups")


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = cast(tuple[int], conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone())
    return row[0]


def _seed_coordinator_children(conn: sqlite3.Connection, session_id: str) -> None:
    # Seed reset tables without starting a worker runtime.
    statements = (
        ("coordinator_links (session_id, kind, entity_id)", (session_id, "node", "1")),
        (
            "coordinator_events (session_id, kind, text, entity_kind, entity_id, "
            + "tool_name, status, created_at)",
            (session_id, "test", "event", "node", "1", "test", "ok", "now"),
        ),
        (
            "coordinator_turn_launches (command_id, coordinator_id, scope_kind, scope_id, state)",
            ("turn", session_id, "coordinator", session_id, "submitted"),
        ),
        (
            "coordinator_dispatches (attempt_id, session_id, group_id, node_id, run_id, state)",
            ("attempt", session_id, "group", 1, "run", "started"),
        ),
        (
            "coordinator_action_receipts "
            + "(command_id, session_id, provider_session_id, action_hash, state, created_at)",
            ("action", session_id, "provider", "hash", "accepted", "now"),
        ),
        ("coordinator_plans (operation_id, session_id)", ("plan", session_id)),
        (
            "coordinator_provider_bindings "
            + "(coordinator_id, scope_kind, scope_id, provider_family, provider_session_id)",
            (session_id, "coordinator", session_id, "codex", "provider"),
        ),
        (
            "coordinator_turn_events "
            + "(coordinator_id, provider_family, provider_session_id, "
            + "turn_id, status, recorded_at)",
            (session_id, "codex", "provider", "turn", "confirmed", "now"),
        ),
    )
    for target, values in statements:
        placeholders = ", ".join("?" for _ in values)
        _ = conn.execute(f"INSERT INTO {target} VALUES ({placeholders})", values)


def _seed_remaining_coordinator_rows(conn: sqlite3.Connection, session_id: str) -> None:
    _ = conn.execute(
        "INSERT INTO coordinator_web_receipts "
        + "(command_id, session_id, command_hash, status, result_json) VALUES (?, ?, ?, ?, ?)",
        ("web", session_id, "hash", "accepted", "{}"),
    )
    _ = conn.execute(
        "INSERT INTO coordinator_plan_proposals "
        + "(id, session_id, manifest_json, context_path, graph_revision, status) "
        + "VALUES (?, ?, ?, ?, ?, ?)",
        ("proposal", session_id, "{}", "context", 0, "pending"),
    )


def test_drop_all_clears_coordinator_rows_and_reuses_command_id(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    control = CoordinatorControl(graph, tmp_path)
    first = control.send_coordinator_command("", StartGoal("same", "First", "codex"))
    assert first.status == "accepted"
    session_id = cast(str, cast(dict[str, object], first.result)["id"])
    conn = graph.group_connection
    _seed_coordinator_children(conn, session_id)
    _seed_remaining_coordinator_rows(conn, session_id)
    conn.commit()
    assert cast(tuple[int], conn.execute("PRAGMA foreign_keys").fetchone())[0] == 1
    assert all(_count(conn, table) > 0 for table in COORDINATOR_TABLES)
    original_count = len(graph.get_all_nodes())

    assert graph.drop_all() == original_count
    assert all(_count(conn, table) == 0 for table in COORDINATOR_TABLES)
    assert cast(tuple[int], conn.execute("PRAGMA foreign_keys").fetchone())[0] == 1
    assert graph.drop_all() == 0
    graph.close()

    reopened = MikadoGraph(path)
    reopened_conn = reopened.group_connection
    assert cast(tuple[int], reopened_conn.execute("PRAGMA foreign_keys").fetchone())[0] == 1
    version = cast(tuple[int], reopened_conn.execute("PRAGMA user_version").fetchone())
    assert version[0] == SCHEMA_VERSION
    new_control = CoordinatorControl(reopened, tmp_path)
    second = new_control.send_coordinator_command("", StartGoal("same", "Second", "codex"))
    assert second.status == "accepted"
    assert second.result != first.result
    assert _count(reopened.group_connection, "coordinator_web_receipts") == 1
    reopened.close()


def test_drop_all_clears_forked_execution_groups_before_nodes(graph: MikadoGraph) -> None:
    source_task = graph.add_node("source")
    source = graph.groups.create(
        "source-graph",
        (source_task.id,),
        GroupWorkspace("/tmp/reset-source", "reset-source", "source-session"),
    )
    fork = graph.groups.fork(
        source.id,
        GroupWorkspace("/tmp/reset-fork", "reset-fork", "fork-session"),
    )
    assert fork.id != source.id
    conn = graph.group_connection
    assert all(_count(conn, table) > 0 for table in GROUP_TABLES)
    original_count = len(graph.get_all_nodes())

    assert graph.drop_all() == original_count
    assert all(_count(conn, table) == 0 for table in GROUP_TABLES)
    assert graph.get_all_nodes() == []
    assert graph.drop_all() == 0
