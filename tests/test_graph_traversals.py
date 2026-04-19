from __future__ import annotations

import pytest

from milknado.domains.graph import MikadoGraph
from milknado.domains.graph.traversals import walk_ancestors


class TestWalkAncestors:
    def test_single_node_returns_just_that_node(self, graph: MikadoGraph) -> None:
        node = graph.add_node("root only")
        result = walk_ancestors(graph, node.id)
        assert result == [node]

    def test_linear_chain_returns_leaf_to_root(self, graph: MikadoGraph) -> None:
        # A is root, B is child of A, C is child of B
        a = graph.add_node("A")
        b = graph.add_node("B", parent_id=a.id)
        c = graph.add_node("C", parent_id=b.id)

        result = walk_ancestors(graph, c.id)
        assert [n.id for n in result] == [c.id, b.id, a.id]

    def test_diamond_returns_deterministic_path(self, graph: MikadoGraph) -> None:
        # D is root; B and C both have D as parent; A has both B and C as parents
        # Structure: D -> B -> A, D -> C -> A
        d = graph.add_node("D")
        b = graph.add_node("B", parent_id=d.id)
        c = graph.add_node("C", parent_id=d.id)
        # A depends on both B and C — add A then manually add second parent edge
        a = graph.add_node("A", parent_id=b.id)
        graph.add_edge(c.id, a.id)

        result = walk_ancestors(graph, a.id)
        # Must start at A and end at D
        assert result[0].id == a.id
        assert result[-1].id == d.id
        # The second element must be a direct parent of A (either B or C)
        assert result[1].id in {b.id, c.id}
        # All nodes reachable on the chosen path must be present
        assert len(result) == 3

    def test_unknown_node_id_raises_value_error(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="Node 9999 not found"):
            walk_ancestors(graph, 9999)
