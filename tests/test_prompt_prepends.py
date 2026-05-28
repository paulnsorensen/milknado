"""Pure-function tests for the prepend params on build_planning_context and render_brief."""

from __future__ import annotations

from pathlib import Path

from milknado.domains.dispatch.brief import render_brief
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
