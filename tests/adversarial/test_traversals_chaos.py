"""Adversarial tests for walk_ancestors.

Focus: cycles (via raw SQL if possible), deep chains, self-parent.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from milknado.domains.graph import MikadoGraph
from milknado.domains.graph.traversals import walk_ancestors


@pytest.fixture()
def graph(tmp_path: Path) -> Generator[MikadoGraph, None, None]:
    g = MikadoGraph(tmp_path / "test.db")
    yield g
    g.close()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


class TestCycleHandling:
    def test_self_parent_does_not_infinite_loop(self, db_path: Path) -> None:
        """Force a self-referential parent_id via raw SQL — walk_ancestors must terminate."""
        g = MikadoGraph(db_path)
        node = g.add_node("self-referential node")
        node_id = node.id
        g.close()

        # Force parent_id = self via raw SQL, bypassing the graph API
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE nodes SET parent_id = ? WHERE id = ?",
                (node_id, node_id),
            )
            conn.commit()

        g2 = MikadoGraph(db_path)
        try:
            result = walk_ancestors(g2, node_id)
            # Cycle guard should break after 1 step — result is just [node]
            assert len(result) >= 1
            assert result[0].id == node_id
        finally:
            g2.close()

    def test_two_node_cycle_terminates(self, db_path: Path) -> None:
        """A→B→A cycle via raw SQL — walk_ancestors must not infinite loop."""
        g = MikadoGraph(db_path)
        a = g.add_node("A")
        b = g.add_node("B", parent_id=a.id)
        g.close()

        # Make A point back to B (cycle: A→B→A)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE nodes SET parent_id = ? WHERE id = ?",
                (b.id, a.id),
            )
            conn.commit()

        g2 = MikadoGraph(db_path)
        try:
            # Starting from B: B→A→B (cycle) — visited set breaks the loop
            result = walk_ancestors(g2, b.id)
            assert len(result) >= 1
            # No node appears twice
            ids = [n.id for n in result]
            assert len(ids) == len(set(ids)), "Cycle produced duplicate nodes in result"
        finally:
            g2.close()


class TestDeepChain:
    def test_chain_100_nodes_terminates(self, graph: MikadoGraph) -> None:
        """100-node linear chain — should not stack overflow or OOM."""
        nodes = [graph.add_node("root")]
        for i in range(1, 100):
            nodes.append(graph.add_node(f"node_{i}", parent_id=nodes[-1].id))

        leaf = nodes[-1]
        result = walk_ancestors(graph, leaf.id)
        assert len(result) == 100
        assert result[0].id == leaf.id
        assert result[-1].id == nodes[0].id

    def test_chain_1000_nodes_no_stack_overflow(self, graph: MikadoGraph) -> None:
        """1000-node chain — iterative implementation should handle this without recursion limit."""
        root = graph.add_node("root")
        current_id = root.id
        for i in range(999):
            node = graph.add_node(f"node_{i}", parent_id=current_id)
            current_id = node.id

        # walk_ancestors is iterative (while loop), so recursion limit shouldn't matter
        result = walk_ancestors(graph, current_id)
        assert len(result) == 1000
        assert result[-1].id == root.id


class TestMissingNode:
    def test_nonexistent_node_raises_value_error(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="9999"):
            walk_ancestors(graph, 9999)

    def test_negative_node_id_raises_value_error(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError):
            walk_ancestors(graph, -1)

    def test_zero_node_id_raises_value_error(self, graph: MikadoGraph) -> None:
        # Node IDs start at 1 in SQLite autoincrement; 0 should not exist.
        with pytest.raises(ValueError):
            walk_ancestors(graph, 0)


class TestDanglingParentRef:
    def test_parent_id_pointing_to_deleted_node_terminates(self, db_path: Path) -> None:
        """parent_id points to a node that was deleted — walk_ancestors must handle gracefully."""
        g = MikadoGraph(db_path)
        a = g.add_node("A")
        b = g.add_node("B", parent_id=a.id)
        b_id = b.id
        g.close()

        # Delete node A via raw SQL, leaving B with a dangling parent_id
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM nodes WHERE id = ?", (a.id,))
            conn.commit()

        g2 = MikadoGraph(db_path)
        try:
            result = walk_ancestors(g2, b_id)
            # get_node(a.id) returns None → walk_ancestors breaks the loop
            assert len(result) == 1
            assert result[0].id == b_id
        finally:
            g2.close()
