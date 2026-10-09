from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from milknado.domains.graph import MikadoGraph
from tests.graph_helpers import graph_conn


def test_standalone_mutations_commit_to_other_connection(graph: MikadoGraph) -> None:
    parent = graph.add_node("parent")
    child = graph.add_node("child")
    with closing(sqlite3.connect(graph.db_path)) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes").fetchone() == (2,)

        _ = graph.add_edge(parent.id, child.id)
        assert observer.execute("SELECT parent_id, child_id FROM edges").fetchone() == (
            parent.id,
            child.id,
        )

        graph.set_parent_id(child.id, parent.id)
        assert observer.execute(
            "SELECT parent_id FROM nodes WHERE id = ?", (child.id,)
        ).fetchone() == (parent.id,)


def test_standalone_failure_rolls_back_partial_node(graph: MikadoGraph) -> None:
    conn = graph_conn(graph)
    with pytest.raises(sqlite3.IntegrityError):
        _ = graph.add_node("failed", files=("src/a.py", "src/a.py"))

    assert not conn.in_transaction
    with closing(sqlite3.connect(graph.db_path)) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes").fetchone() == (0,)
        assert observer.execute("SELECT COUNT(*) FROM file_ownership").fetchone() == (0,)


def test_caller_owns_successful_nested_mutations(graph: MikadoGraph) -> None:
    conn = graph_conn(graph)
    parent = graph.add_node("parent")
    child = graph.add_node("child")

    _ = conn.execute("BEGIN IMMEDIATE")
    nested = graph.add_node("nested")
    assert conn.in_transaction
    _ = graph.add_edge(parent.id, child.id)
    assert conn.in_transaction
    graph.set_parent_id(child.id, parent.id)
    assert conn.in_transaction

    with closing(sqlite3.connect(graph.db_path)) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes").fetchone() == (2,)
        assert observer.execute("SELECT COUNT(*) FROM edges").fetchone() == (0,)
        assert observer.execute(
            "SELECT parent_id FROM nodes WHERE id = ?", (child.id,)
        ).fetchone() == (None,)

    conn.rollback()
    assert graph.get_node(nested.id) is None
    refreshed = graph.get_node(child.id)
    assert refreshed is not None
    assert refreshed.parent_id is None
    with closing(sqlite3.connect(graph.db_path)) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes").fetchone() == (2,)
        assert observer.execute("SELECT COUNT(*) FROM edges").fetchone() == (0,)


def test_nested_failure_preserves_caller_work_and_transaction(graph: MikadoGraph) -> None:
    conn = graph_conn(graph)
    parent = graph.add_node("parent")
    child = graph.add_node("child")

    _ = conn.execute("BEGIN IMMEDIATE")
    pending = graph.add_node("pending")
    with pytest.raises(sqlite3.IntegrityError):
        _ = graph.add_node("failed", files=("src/a.py", "src/a.py"))
    assert conn.in_transaction
    with pytest.raises(ValueError, match="would create a cycle"):
        _ = graph.add_edge(parent.id, parent.id)
    assert conn.in_transaction
    with pytest.raises(ValueError, match="not found"):
        graph.set_parent_id(child.id, 999999)
    assert conn.in_transaction
    assert graph.get_node(pending.id) is not None

    with closing(sqlite3.connect(graph.db_path)) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes").fetchone() == (2,)
    conn.rollback()
    with closing(sqlite3.connect(graph.db_path)) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes").fetchone() == (2,)
