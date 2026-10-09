from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

from milknado.domains.graph import MikadoGraph
from milknado.domains.graph._persistence import SCHEMA_VERSION

CORE_TABLES = {
    "coordinator_sessions",
    "coordinator_links",
    "coordinator_events",
    "coordinator_provider_bindings",
    "coordinator_turn_events",
    "coordinator_dispatches",
    "coordinator_action_receipts",
    "coordinator_plans",
}
CORE_INDEXES = {
    "idx_coordinator_events_session",
    "idx_coordinator_events_expiry",
    "idx_coordinator_events_operation",
    "idx_coordinator_turn_unknown",
}


def test_fresh_graph_owns_complete_coordinator_schema(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    graph.close()
    with closing(sqlite3.connect(path)) as conn:
        objects = set(
            cast(
                list[tuple[str, str]],
                conn.execute(
                    "SELECT type, name FROM sqlite_master WHERE name LIKE 'coordinator_%' "
                    + "OR name LIKE 'idx_coordinator_%'"
                ).fetchall(),
            )
        )
        assert {("table", name) for name in CORE_TABLES} <= objects
        assert {("index", name) for name in CORE_INDEXES} <= objects
        columns = {
            row[1]
            for row in cast(
                list[tuple[object, str, str, int, object, int]],
                conn.execute("PRAGMA table_info(coordinator_events)").fetchall(),
            )
        }
        assert "operation_hash" in columns


def test_current_coordinator_schema_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    MikadoGraph(path).close()
    MikadoGraph(path).close()
    with closing(sqlite3.connect(path)) as conn:
        columns = {
            row[1]
            for row in cast(
                list[tuple[object, str, str, int, object, int]],
                conn.execute("PRAGMA table_info(coordinator_events)").fetchall(),
            )
        }
        assert "operation_hash" in columns
        version = cast(tuple[int], conn.execute("PRAGMA user_version").fetchone())[0]
        assert version == SCHEMA_VERSION
