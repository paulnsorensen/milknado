"""Pure-function tests for the prepend params on build_planning_context and render_brief."""

from __future__ import annotations

from pathlib import Path

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.dispatch import render_brief
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning.context import build_planning_context


def test_build_planning_context_prepends_when_set(tmp_path: Path, mock_crg) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    try:
        ctx = build_planning_context(
            "ship it",
            mock_crg,
            graph,
            prepend="**TEAM NOTE:** read AGENTS.md first.",
        )
    finally:
        graph.close()
    # Prepend appears before the # Goal header.
    assert ctx.startswith("**TEAM NOTE:** read AGENTS.md first.")
    assert ctx.index("**TEAM NOTE:**") < ctx.index("# Goal")


def test_build_planning_context_omits_prepend_when_none(tmp_path: Path, mock_crg) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    try:
        ctx = build_planning_context("ship it", mock_crg, graph)
    finally:
        graph.close()
    assert ctx.startswith("# Goal")


def test_render_brief_prepends_when_set(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    try:
        node = graph.add_node("do thing")
        node_id = node.id
        brief = render_brief(graph, node_id, prepend="### House rule: run just build.")
    finally:
        graph.close()
    assert brief.startswith("### House rule: run just build.")
    # Original task heading still present after the prepend.
    assert "# Task: do thing" in brief


def test_render_brief_omits_prepend_when_none(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    try:
        node = graph.add_node("do thing")
        node_id = node.id
        brief = render_brief(graph, node_id)
    finally:
        graph.close()
    assert brief.startswith("# Task: do thing")


def test_render_brief_review_flavor_uses_review_instructions(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    try:
        node = graph.add_node("review the diff", spec=NodeSpec(flavor="review"))
        brief = render_brief(graph, node.id)
    finally:
        graph.close()
    assert brief.startswith("# Review: review the diff")
    assert "## Work under review" in brief
    assert "severity-grouped findings report" in brief
    assert "Relevant files" not in brief


def test_render_brief_plate_flavor_uses_plate_instructions(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    try:
        node = graph.add_node("ship it", spec=NodeSpec(flavor="plate"))
        brief = render_brief(graph, node.id)
    finally:
        graph.close()
    assert brief.startswith("# Plate: ship it")
    assert "## Work to publish" in brief
    assert "terminal plating task" in brief
    assert "`gh` is authenticated in this environment without" in brief


def test_render_brief_embeds_own_spec_path(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    try:
        node = graph.add_node("do thing", spec=NodeSpec(artifact_path=".cheese/specs/foo.md"))
        brief = render_brief(graph, node.id)
    finally:
        graph.close()
    assert "## Spec" in brief
    assert "- .cheese/specs/foo.md" in brief


def test_render_brief_embeds_ancestor_goal_spec_path(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    try:
        goal = graph.add_node(
            "ship the feature",
            spec=NodeSpec(kind=NodeKind.GOAL, artifact_path=".cheese/specs/feature.md"),
        )
        task = graph.add_node("do the subtask", parent_id=goal.id)
        brief = render_brief(graph, task.id)
    finally:
        graph.close()
    assert "## Spec" in brief
    assert "- .cheese/specs/feature.md" in brief


def test_render_brief_omits_spec_section_when_absent(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    try:
        node = graph.add_node("do thing")
        brief = render_brief(graph, node.id)
    finally:
        graph.close()
    assert "## Spec" not in brief
