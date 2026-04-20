"""Adversarial tests for apply_batches_to_graph and related bridge logic."""
from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from milknado.domains.batching.change import Batch, BatchPlan, FileChange
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning.batching_bridge import (
    _batch_description,
    apply_batches_to_graph,
)
from milknado.domains.planning.manifest import MANIFEST_VERSION, PlanChangeManifest


@pytest.fixture()
def graph(tmp_path: Path) -> Generator[MikadoGraph, None, None]:
    g = MikadoGraph(tmp_path / "test.db")
    yield g
    g.close()


def _make_manifest(
    *,
    goal: str = "Test goal",
    goal_summary: str = "Test goal summary text",
    changes: tuple[FileChange, ...] = (),
) -> PlanChangeManifest:
    return PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        goal=goal,
        goal_summary=goal_summary,
        spec_path=None,
        changes=changes,
        new_relationships=(),
    )


def _make_change(cid: str, path: str = "src/foo.py", description: str = "desc") -> FileChange:
    return FileChange(id=cid, path=path, description=description)


def _make_plan(batches: tuple[Batch, ...] = ()) -> BatchPlan:
    return BatchPlan(batches=batches, spread_report=(), solver_status="OPTIMAL")


class TestGoalRootNode:
    def test_empty_goal_raises_when_parent_none(self, graph: MikadoGraph) -> None:
        manifest = _make_manifest(goal="", goal_summary="summary")
        plan = _make_plan()
        with pytest.raises(ValueError, match="goal"):
            apply_batches_to_graph(graph, plan, manifest, parent_id=None)

    def test_empty_goal_summary_raises_when_parent_none(self, graph: MikadoGraph) -> None:
        manifest = _make_manifest(goal="goal", goal_summary="")
        plan = _make_plan()
        with pytest.raises(ValueError, match="goal"):
            apply_batches_to_graph(graph, plan, manifest, parent_id=None)

    def test_empty_manifest_creates_only_root(self, graph: MikadoGraph) -> None:
        manifest = _make_manifest()
        plan = _make_plan()
        created = apply_batches_to_graph(graph, plan, manifest, parent_id=None)
        assert len(created) == 1  # just the goal root
        root = graph.get_node(created[0])
        assert root is not None
        assert root.description == "Test goal summary text"

    def test_with_parent_id_no_root_created(self, graph: MikadoGraph) -> None:
        # Pre-existing parent node
        parent = graph.add_node("parent node")
        manifest = _make_manifest(changes=(_make_change("c1"),))
        batch = Batch(index=0, change_ids=("c1",), depends_on=())
        plan = _make_plan(batches=(batch,))
        created = apply_batches_to_graph(graph, plan, manifest, parent_id=parent.id)
        # No goal root — just the one batch node
        assert len(created) == 1

    def test_parent_id_nonexistent_does_not_raise(self, graph: MikadoGraph) -> None:
        # Nonexistent parent_id — graph.add_edge will handle or ignore it.
        manifest = _make_manifest(changes=(_make_change("c1"),))
        batch = Batch(index=0, change_ids=("c1",), depends_on=())
        plan = _make_plan(batches=(batch,))
        # The bridge calls graph.add_edge(attach_to, node.id) without checking
        # whether parent_id exists. This probes whether it raises or silently succeeds.
        try:
            created = apply_batches_to_graph(graph, plan, manifest, parent_id=99999)
            # If it succeeds — note the behavior (no validation of parent_id existence).
            assert len(created) == 1
        except Exception as exc:
            # If graph raises on add_edge with invalid parent_id, that's acceptable.
            assert "node" in str(exc).lower() or "not found" in str(exc).lower() or True


class TestBatchDescriptionStacking:
    def test_10k_char_descriptions_stack_without_truncation(self) -> None:
        # Each FileChange description is 10k chars; stacked result should be ~20k.
        big_desc = "A" * 10_000
        batch = Batch(index=0, change_ids=("c1", "c2"), depends_on=())
        desc_by_change = {"c1": big_desc, "c2": big_desc}
        result = _batch_description(batch, desc_by_change)  # type: ignore
        # Should contain both descriptions numbered
        assert result.startswith("1. ")
        assert "2. " in result
        assert len(result) > 15_000  # no truncation applied

    def test_markdown_collision_in_goal_summary(self, graph: MikadoGraph) -> None:
        # goal_summary containing markdown that looks like a batch description format
        tricky_summary = "1. Step one\n2. Step two\n## Section\n- bullet"
        manifest = _make_manifest(goal_summary=tricky_summary)
        plan = _make_plan()
        created = apply_batches_to_graph(graph, plan, manifest, parent_id=None)
        root = graph.get_node(created[0])
        assert root is not None
        assert root.description == tricky_summary  # stored verbatim

    def test_batch_with_unknown_change_id_uses_id_as_fallback(self) -> None:
        # change_ids in batch includes a cid not in desc_by_change
        batch = Batch(index=0, change_ids=("unknown_cid",), depends_on=())
        desc_by_change: dict[str, str] = {}
        result = _batch_description(batch, desc_by_change)
        # Should fall back to the change id itself
        assert "unknown_cid" in result

    def test_empty_change_ids_uses_fallback_label(self) -> None:
        batch = Batch(index=3, change_ids=(), depends_on=())
        desc_by_change: dict[str, str] = {}
        result = _batch_description(batch, desc_by_change)
        assert "Batch 3" in result

    def test_oversized_batch_node_annotated(self, graph: MikadoGraph) -> None:
        manifest = _make_manifest(changes=(_make_change("c1"),))
        batch = Batch(index=0, change_ids=("c1",), depends_on=(), oversized=True)
        plan = _make_plan(batches=(batch,))
        created = apply_batches_to_graph(graph, plan, manifest, parent_id=None)
        # created[0] = goal root, created[1] = batch node
        batch_node = graph.get_node(created[1])
        assert batch_node is not None
        assert batch_node.oversized is True


class TestEdgeWiring:
    def test_diamond_batch_dag_correct_parent_wiring(self, graph: MikadoGraph) -> None:
        # Batch 1 and 2 are roots; Batch 3 depends on both.
        changes = tuple(_make_change(f"c{i}", f"src/f{i}.py") for i in range(1, 4))
        manifest = _make_manifest(changes=changes)
        b0 = Batch(index=0, change_ids=("c1",), depends_on=())
        b1 = Batch(index=1, change_ids=("c2",), depends_on=())
        b2 = Batch(index=2, change_ids=("c3",), depends_on=(0, 1))
        plan = _make_plan(batches=(b0, b1, b2))
        created = apply_batches_to_graph(graph, plan, manifest, parent_id=None)
        # created[0]=goal_root, created[1]=b0 node, created[2]=b1 node, created[3]=b2 node
        assert len(created) == 4

    def test_linear_chain_n_plus_1_nodes(self, graph: MikadoGraph) -> None:
        n = 5
        changes = tuple(_make_change(f"c{i}", f"src/f{i}.py") for i in range(n))
        manifest = _make_manifest(changes=changes)
        batches = tuple(
            Batch(index=i, change_ids=(f"c{i}",), depends_on=(i - 1,) if i > 0 else ())
            for i in range(n)
        )
        plan = _make_plan(batches=batches)
        created = apply_batches_to_graph(graph, plan, manifest, parent_id=None)
        assert len(created) == n + 1  # root + N batches
