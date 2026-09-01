"""github_ref column + graph API: the durable node-id key for the GitHub
Projects v2 <-> milknado roadmap crossover.

Mirrors the wiki_ref tests — pins the additive nullable column + partial UNIQUE
index and the two graph methods the importer/binder need: writing a github_ref
on create and finding a node by it.
"""

from __future__ import annotations

# These tests intentionally inspect private graph persistence state.
import sqlite3
from typing import cast

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.graph import MikadoGraph
from tests.graph_helpers import graph_conn


class TestGithubRefColumn:
    def test_fresh_db_has_github_ref_column_and_index(self, graph: MikadoGraph) -> None:
        conn = graph_conn(graph)
        cols = {
            row[1]
            for row in cast(
                list[tuple[object, str]], conn.execute("PRAGMA table_info(nodes)").fetchall()
            )
        }
        assert "github_ref" in cols
        idx = {
            row[1]
            for row in cast(
                list[tuple[object, str]], conn.execute("PRAGMA index_list(nodes)").fetchall()
            )
        }
        assert "idx_nodes_github_ref" in idx


class TestGithubRefGraphApi:
    def test_add_node_persists_github_ref(self, graph: MikadoGraph) -> None:
        node = graph.add_node("roadmap", spec=NodeSpec(kind=NodeKind.ROADMAP, github_ref="PVT_1"))
        assert node.github_ref == "PVT_1"
        reloaded = graph.get_node(node.id)
        assert reloaded is not None
        assert reloaded.github_ref == "PVT_1"

    def test_add_node_github_ref_defaults_none(self, graph: MikadoGraph) -> None:
        assert graph.add_node("plain").github_ref is None

    def test_find_node_by_github_ref_returns_match(self, graph: MikadoGraph) -> None:
        created = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL, github_ref="PVTI_x"))
        found = graph.find_node_by_github_ref("PVTI_x")
        assert found is not None
        assert found.id == created.id

    def test_find_node_by_github_ref_missing_returns_none(self, graph: MikadoGraph) -> None:
        _ = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL, github_ref="PVTI_present"))
        assert graph.find_node_by_github_ref("PVTI_absent") is None

    def test_unique_index_rejects_duplicate_github_ref(self, graph: MikadoGraph) -> None:
        _ = graph.add_node("first", spec=NodeSpec(kind=NodeKind.GOAL, github_ref="dup"))
        with pytest.raises(sqlite3.IntegrityError):
            _ = graph.add_node("second", spec=NodeSpec(kind=NodeKind.GOAL, github_ref="dup"))

    def test_multiple_null_github_refs_allowed(self, graph: MikadoGraph) -> None:
        # Partial index must not treat NULL as a colliding value.
        _ = graph.add_node("a")
        _ = graph.add_node("b")
        assert len([n for n in graph.get_all_nodes() if n.github_ref is None]) == 2

    def test_set_github_ref_attaches_key(self, graph: MikadoGraph) -> None:
        node = graph.add_node("orphan goal", spec=NodeSpec(kind=NodeKind.GOAL))
        assert node.github_ref is None
        graph.set_github_ref(node.id, "PVTI_late")
        assert graph.find_node_by_github_ref("PVTI_late") is not None

    def test_set_github_ref_unknown_node_raises(self, graph: MikadoGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.set_github_ref(999, "x")
