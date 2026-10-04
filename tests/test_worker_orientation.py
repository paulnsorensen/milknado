"""The worker brief states run id, node id, worktree and branch up front."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.dispatch import WorkerOrientation, render_brief
from milknado.domains.graph import MikadoGraph

ORIENTATION = WorkerOrientation(
    run_id="run-1-abc", worktree=Path("/work/tree"), branch="milknado/node-1"
)


def _graph_with_node(tmp_path: Path, flavor: str | None = None) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(tmp_path / "g.db")
    node = graph.add_node("do the thing", spec=NodeSpec(kind=NodeKind.TASK, flavor=flavor))
    return graph, node.id


@pytest.mark.parametrize("flavor", [None, "review", "plate"])
def test_brief_states_orientation_for_every_flavor(tmp_path: Path, flavor: str | None) -> None:
    graph, node_id = _graph_with_node(tmp_path, flavor)
    try:
        brief = render_brief(graph, node_id, orientation=ORIENTATION)
    finally:
        graph.close()
    assert "- run_id: run-1-abc" in brief
    assert f"- node_id: {node_id}" in brief
    assert "- worktree: /work/tree" in brief
    assert "- branch: milknado/node-1" in brief
    assert "run_id stated under Orientation" in brief
    assert "MILKNADO_RUN_ID" not in brief


def test_orientation_follows_prepend_and_precedes_goal_context(tmp_path: Path) -> None:
    graph, node_id = _graph_with_node(tmp_path)
    try:
        brief = render_brief(graph, node_id, prepend="TEAM NOTE", orientation=ORIENTATION)
    finally:
        graph.close()
    order = [brief.index(m) for m in ("TEAM NOTE", "## Orientation", "## Goal context")]
    assert order == sorted(order)


def test_unknown_branch_is_stated_as_unknown(tmp_path: Path) -> None:
    graph, node_id = _graph_with_node(tmp_path)
    try:
        brief = render_brief(graph, node_id, orientation=WorkerOrientation("r", Path("/w"), None))
    finally:
        graph.close()
    assert "- branch: (unknown)" in brief


def test_brief_without_orientation_has_no_orientation_block(tmp_path: Path) -> None:
    graph, node_id = _graph_with_node(tmp_path)
    try:
        brief = render_brief(graph, node_id)
    finally:
        graph.close()
    assert "## Orientation" not in brief


@pytest.mark.parametrize("flavor", [None, "review", "plate"])
def test_brief_without_orientation_names_the_run_id_environment_variable(
    tmp_path: Path, flavor: str | None
) -> None:
    graph, node_id = _graph_with_node(tmp_path, flavor)
    try:
        brief = render_brief(graph, node_id)
    finally:
        graph.close()
    assert "run_id set to the MILKNADO_RUN_ID environment variable" in brief
    assert "stated under Orientation" not in brief
