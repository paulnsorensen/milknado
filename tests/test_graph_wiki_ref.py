"""wiki_ref column + graph API: the durable content-addressed key for the
wiki <-> milknado roadmap crossover.

These tests pin the additive DB change (nullable wiki_ref + partial UNIQUE
index riding the auto-migrate-on-load path) and the two graph methods the
importer/exporter need: writing a wiki_ref on create and finding a node by it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from milknado.domains.common import NodeKind
from milknado.domains.graph import MikadoGraph
from milknado.domains.graph._persistence import ensure_schema, row_to_node


class TestWikiRefColumn:
    def test_fresh_db_has_wiki_ref_column_and_index(self, graph: MikadoGraph) -> None:
        conn = graph._conn
        cols = {row[1] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()}
        assert "wiki_ref" in cols
        idx = {row[1] for row in conn.execute("PRAGMA index_list(nodes)").fetchall()}
        assert "idx_nodes_wiki_ref" in idx

    def test_ensure_schema_adds_wiki_ref_to_legacy_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        c.executescript("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parent_id INTEGER,
                created_at TEXT NOT NULL
            );
        """)
        c.commit()
        ensure_schema(c)
        cols = {row[1] for row in c.execute("PRAGMA table_info(nodes)").fetchall()}
        assert "wiki_ref" in cols
        idx = {row[1] for row in c.execute("PRAGMA index_list(nodes)").fetchall()}
        assert "idx_nodes_wiki_ref" in idx
        c.close()

    def test_row_to_node_defaults_wiki_ref_when_column_absent(self, tmp_path: Path) -> None:
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
            ("sparse", "pending", "2026-01-01T00:00:00+00:00"),
        )
        c.commit()
        row = c.execute("SELECT * FROM nodes WHERE id = 1").fetchone()
        node = row_to_node(row)
        assert node.wiki_ref is None
        c.close()


class TestWikiRefGraphApi:
    def test_add_node_persists_wiki_ref(self, graph: MikadoGraph) -> None:
        node = graph.add_node("roadmap", kind=NodeKind.ROADMAP, wiki_ref="abc-123")
        assert node.wiki_ref == "abc-123"
        reloaded = graph.get_node(node.id)
        assert reloaded is not None
        assert reloaded.wiki_ref == "abc-123"

    def test_add_node_wiki_ref_defaults_none(self, graph: MikadoGraph) -> None:
        assert graph.add_node("plain").wiki_ref is None

    def test_find_node_by_wiki_ref_returns_match(self, graph: MikadoGraph) -> None:
        created = graph.add_node("goal", kind=NodeKind.GOAL, wiki_ref="ref-xyz")
        found = graph.find_node_by_wiki_ref("ref-xyz")
        assert found is not None
        assert found.id == created.id

    def test_find_node_by_wiki_ref_missing_returns_none(self, graph: MikadoGraph) -> None:
        graph.add_node("goal", kind=NodeKind.GOAL, wiki_ref="ref-present")
        assert graph.find_node_by_wiki_ref("ref-absent") is None

    def test_unique_index_rejects_duplicate_wiki_ref(self, graph: MikadoGraph) -> None:
        graph.add_node("first", kind=NodeKind.GOAL, wiki_ref="dup")
        with pytest.raises(sqlite3.IntegrityError):
            graph.add_node("second", kind=NodeKind.GOAL, wiki_ref="dup")

    def test_multiple_null_wiki_refs_allowed(self, graph: MikadoGraph) -> None:
        # Partial index must not treat NULL as a colliding value.
        graph.add_node("a")
        graph.add_node("b")
        assert len([n for n in graph.get_all_nodes() if n.wiki_ref is None]) == 2

    def test_set_wiki_ref_attaches_key(self, graph: MikadoGraph) -> None:
        node = graph.add_node("orphan goal", kind=NodeKind.GOAL)
        assert node.wiki_ref is None
        graph.set_wiki_ref(node.id, "late-ref")
        assert graph.find_node_by_wiki_ref("late-ref") is not None

    def test_set_wiki_ref_unknown_node_raises(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.set_wiki_ref(999, "x")
