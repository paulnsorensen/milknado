"""Tests for graph/_persistence.py — schema, row serialization, plan_state, drop_all."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from milknado.domains.graph import MikadoGraph
from milknado.domains.graph._persistence import (
    create_tables,
    ensure_schema,
    get_spec_hash,
    row_to_node,
    set_dispatched_at,
    set_spec_hash,
)


@pytest.fixture()
def conn(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    c = sqlite3.connect(str(tmp_path / "p.db"))
    c.row_factory = sqlite3.Row
    create_tables(c)
    ensure_schema(c)
    yield c
    c.close()


@pytest.fixture()
def graph(tmp_path: Path) -> Generator[MikadoGraph, None, None]:
    g = MikadoGraph(tmp_path / "g.db")
    yield g
    g.close()


class TestCreateTables:
    def test_tables_created(self, conn: sqlite3.Connection) -> None:
        names = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {"nodes", "edges", "file_ownership", "plan_state", "batch_plans"} <= names


class TestPlanState:
    def test_set_and_get_spec_hash(self, conn: sqlite3.Connection) -> None:
        set_spec_hash(conn, "abc123")
        assert get_spec_hash(conn) == "abc123"

    def test_get_spec_hash_empty_returns_none(self, conn: sqlite3.Connection) -> None:
        assert get_spec_hash(conn) is None

    def test_set_spec_hash_upserts(self, conn: sqlite3.Connection) -> None:
        set_spec_hash(conn, "first")
        set_spec_hash(conn, "second")
        assert get_spec_hash(conn) == "second"

    def test_graph_set_get_spec_hash(self, graph: MikadoGraph) -> None:
        graph.set_spec_hash("deadbeef")
        assert graph.get_spec_hash() == "deadbeef"

    def test_graph_get_spec_hash_none_before_set(self, graph: MikadoGraph) -> None:
        assert graph.get_spec_hash() is None


class TestDropAll:
    def test_drop_all_returns_node_count(self, graph: MikadoGraph) -> None:
        graph.add_node("one")
        graph.add_node("two")
        count = graph.drop_all()
        assert count == 2

    def test_drop_all_clears_nodes(self, graph: MikadoGraph) -> None:
        graph.add_node("node")
        graph.drop_all()
        assert graph.get_all_nodes() == []

    def test_drop_all_clears_plan_state(self, graph: MikadoGraph) -> None:
        graph.set_spec_hash("abc")
        graph.drop_all()
        assert graph.get_spec_hash() is None

    def test_drop_all_on_empty_returns_zero(self, graph: MikadoGraph) -> None:
        assert graph.drop_all() == 0


class TestSetDispatchedAt:
    def test_set_dispatched_at_raises_for_missing_node(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            set_dispatched_at(conn, 9999)

    def test_set_dispatched_at_persists(self, graph: MikadoGraph) -> None:
        graph.add_node("task")
        graph.set_dispatched_at(1)
        node = graph.get_node(1)
        assert node is not None
        assert node.dispatched_at is not None


class TestSchemaEvolution:
    def test_ensure_schema_adds_missing_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        c.executescript("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parent_id INTEGER,
                worktree_path TEXT,
                branch_name TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE edges (
                parent_id INTEGER NOT NULL,
                child_id INTEGER NOT NULL,
                PRIMARY KEY (parent_id, child_id)
            );
            CREATE TABLE file_ownership (
                node_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                PRIMARY KEY (node_id, file_path)
            );
        """)
        c.execute(
            "INSERT INTO nodes (description, status, created_at) VALUES (?, ?, ?)",
            ("legacy", "pending", "2026-01-01T00:00:00+00:00"),
        )
        c.commit()
        ensure_schema(c)
        cols = {row[1] for row in c.execute("PRAGMA table_info(nodes)").fetchall()}
        assert "run_id" in cols
        assert "dispatched_at" in cols
        assert "completion_duration_seconds" in cols
        c.close()

    def test_ensure_schema_idempotent(self, conn: sqlite3.Connection) -> None:
        # Running ensure_schema twice should not raise
        ensure_schema(conn)
        ensure_schema(conn)


class TestRowToNode:
    def test_handles_missing_optional_columns(self, tmp_path: Path) -> None:
        """row_to_node gracefully handles rows without optional columns."""
        db_path = tmp_path / "sparse.db"
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        c.executescript("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parent_id INTEGER,
                worktree_path TEXT,
                branch_name TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
        """)
        c.execute(
            "INSERT INTO nodes (description, status, created_at) VALUES (?, ?, ?)",
            ("sparse node", "pending", "2026-01-01T00:00:00+00:00"),
        )
        c.commit()
        row = c.execute("SELECT * FROM nodes WHERE id = 1").fetchone()
        node = row_to_node(row)
        assert node.run_id is None
        assert node.dispatched_at is None
        assert node.completion_duration_seconds is None
        c.close()
