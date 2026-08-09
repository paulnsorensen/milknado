from __future__ import annotations

import html as html_lib
from datetime import date

from msgspec import structs

from milknado.domains.common.types import NodeStatus
from milknado.domains.wiki.model import GoalDocument, Lifecycle, RoadmapDocument, RoadmapModel
from milknado.domains.wiki.render import render_html, render_mermaid


def _model() -> RoadmapModel:
    return RoadmapModel(
        roadmap=RoadmapDocument(slug="demo-roadmap", kind="roadmap", created=date(2026, 8, 1)),
        goals={
            "a": GoalDocument(
                slug="a",
                roadmap="demo-roadmap",
                kind="goal",
                created=date(2026, 8, 2),
                status=NodeStatus.DONE,
                title="Goal A",
                intent="Do A.",
                acceptance="- A works.",
            ),
            "b": GoalDocument(
                slug="b",
                roadmap="demo-roadmap",
                kind="goal",
                created=date(2026, 8, 2),
                status=NodeStatus.BLOCKED,
                lifecycle=Lifecycle.DEPRECATED,
                prereqs=["a"],
                title="Goal B",
                intent=None,
                acceptance=None,
            ),
        },
    )


def test_render_mermaid_nodes_edges_and_styles() -> None:
    output = render_mermaid(_model())

    assert output.startswith("graph TD")
    assert 'a["Goal A"]' in output
    assert 'b["Goal B"]' in output
    assert "a --> b" in output
    assert "classDef status-done" in output
    assert "classDef status-blocked" in output
    assert "classDef lifecycle-deprecated" in output
    assert "class b status-blocked,lifecycle-deprecated" in output


def test_render_outputs_are_deterministic_and_html_is_self_contained() -> None:
    model = _model()
    mermaid = render_mermaid(model)
    html = render_html(model)

    assert render_mermaid(model) == mermaid
    assert render_html(model) == html
    assert html_lib.escape(mermaid) in html
    assert "<svg" in html
    assert "stroke-dasharray" in html
    assert html.startswith("<!doctype html>")
    assert "<html" in html
    assert "http://" not in html
    assert "https://" not in html


def test_render_uses_all_edges_and_unique_ids() -> None:
    goals = {
        "a.b": GoalDocument(
            slug="a.b",
            roadmap="demo-roadmap",
            kind="goal",
            created=date(2026, 8, 2),
            title="Dotted",
            intent="Do it.",
            acceptance="Works.",
        ),
        "node_a_b": GoalDocument(
            slug="node_a_b",
            roadmap="demo-roadmap",
            kind="goal",
            created=date(2026, 8, 2),
            title="Underscored",
            intent="Do it.",
            acceptance="Works.",
        ),
        "source": GoalDocument(
            slug="source",
            roadmap="demo-roadmap",
            kind="goal",
            created=date(2026, 8, 2),
            title="Source",
            up=["target"],
            intent="Do it.",
            acceptance="Works.",
        ),
        "target": GoalDocument(
            slug="target",
            roadmap="demo-roadmap",
            kind="goal",
            created=date(2026, 8, 2),
            down=["source"],
            title="Target",
            intent="Do it.",
            acceptance="Works.",
        ),
    }
    model = RoadmapModel(
        roadmap=RoadmapDocument(slug="demo-roadmap", kind="roadmap", created=date(2026, 8, 1)),
        goals=goals,
    )
    output = render_mermaid(model)
    assert 'node_a_b["Dotted"]' in output
    assert 'node_a_b_2["Underscored"]' in output
    assert "source --> target" in output


def test_render_html_escapes_mermaid_source() -> None:
    original = _model()
    goals = {**original.goals, "a": structs.replace(original.goals["a"], title="<script>")}
    model = structs.replace(original, goals=goals)
    output = render_html(model)
    assert "&lt;script&gt;" in output
    assert "<script>" not in output
