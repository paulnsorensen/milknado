from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from milknado.domains.wiki.model import (
    Lifecycle,
    load_roadmap,
    parse_goal_document,
    roadmap_schema,
)

ROADMAP = """---
kind: roadmap
slug: demo-roadmap
created: 2026-08-01
---
# Demo Roadmap
"""

GOAL_A = """---
kind: goal
slug: a
roadmap: demo-roadmap
created: 2026-08-02
status: pending
lifecycle: active
prereqs: []
up: [b]
down: []
last_verified: 2026-08-03
---
# Goal A

## Intent
Do A.

## Acceptance
- A works.

## Outcome
Done A.
"""

GOAL_B = """---
kind: goal
slug: b
roadmap: demo-roadmap
created: 2026-08-02
status: done
prereqs: [a, a]
up: []
down: []
---
# Goal B

## Intent
Do B.

## Acceptance
- B works.
"""


def test_parse_goal_document_and_edges() -> None:
    goal = parse_goal_document(GOAL_A, slug="a")
    assert goal.slug == "a"
    assert goal.title == "Goal A"
    assert goal.created == date(2026, 8, 2)
    assert goal.intent == "Do A."
    assert goal.acceptance == "- A works."
    assert goal.edges == ()


def test_deprecated_goal_may_omit_intent_and_acceptance() -> None:
    text = """---
kind: goal
slug: old
roadmap: demo-roadmap
created: 2026-08-02
lifecycle: deprecated
---
# Old
"""
    goal = parse_goal_document(text, slug="old")
    assert goal.lifecycle is Lifecycle.DEPRECATED
    assert goal.intent is None
    assert goal.acceptance is None


def test_active_goal_requires_intent_and_acceptance() -> None:
    text = """---
kind: goal
slug: active
roadmap: demo-roadmap
created: 2026-08-02
---
# Active
"""
    with pytest.raises(ValueError, match="active.md.*intent|active.md.*acceptance"):
        parse_goal_document(text, slug="active")


@pytest.mark.parametrize("link", ("[[b]]", "[[b|Alias]]"))
def test_goal_wikilinks_resolve_target(link: str) -> None:
    text = GOAL_A.replace("up: [b]", "up: []").replace("down: []", f'down: ["{link}"]')
    goal = parse_goal_document(text, slug="a")
    assert goal.down == ("b",)


def test_load_roadmap_rejects_symlinked_index(tmp_path: Path) -> None:
    directory = tmp_path / "demo-roadmap"
    directory.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text(ROADMAP)
    (directory / "index.md").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinked roadmap path"):
        load_roadmap(directory)


def test_load_roadmap_rejects_symlinked_goal(tmp_path: Path) -> None:
    directory = tmp_path / "demo-roadmap"
    directory.mkdir()
    (directory / "index.md").write_text(ROADMAP)
    outside = tmp_path / "outside.md"
    outside.write_text(GOAL_A)
    (directory / "a.md").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinked roadmap path"):
        load_roadmap(directory)


def test_load_roadmap_and_validate_json_schema(tmp_path: Path) -> None:
    directory = tmp_path / "demo-roadmap"
    directory.mkdir()
    (directory / "index.md").write_text(ROADMAP)
    (directory / "a.md").write_text(
        GOAL_A.replace("up: [b]", "up: []").replace("down: []", "down: [b]")
    )
    (directory / "b.md").write_text(
        GOAL_B.replace("prereqs: [a, a]", "prereqs: []").replace("up: []", "up: [a]")
    )
    model = load_roadmap(directory)
    assert model.roadmap.slug == "demo-roadmap"
    assert set(model.goals) == {"a", "b"}
    assert model.goals["b"].edges == ()
    json.loads(json.dumps(roadmap_schema()))


def test_repository_roadmap_loads() -> None:
    directory = Path(__file__).parents[1] / ".hallouminate/wiki/roadmaps/milestone-0-3-0"
    model = load_roadmap(directory)
    assert model.roadmap.slug == "milestone-0-3-0"
    assert model.goals


def test_bad_edge_target_names_file_and_field(tmp_path: Path) -> None:
    directory = tmp_path / "demo-roadmap"
    directory.mkdir()
    (directory / "index.md").write_text(ROADMAP)
    (directory / "a.md").write_text(
        GOAL_A.replace("up: [b]", "up: []").replace("down: []", "down: [missing]")
    )
    with pytest.raises(ValueError, match=r"a\.md.*down"):
        load_roadmap(directory)


def test_up_down_asymmetry_and_prereq_cycle_fail(tmp_path: Path) -> None:
    directory = tmp_path / "demo-roadmap"
    directory.mkdir()
    (directory / "index.md").write_text(ROADMAP)
    (directory / "a.md").write_text(
        GOAL_A.replace("up: [b]", "up: []").replace("down: []", "down: [b]")
    )
    (directory / "b.md").write_text(
        GOAL_B.replace("prereqs: [a, a]", "prereqs: [a]").replace("up: []", "up: []")
    )
    with pytest.raises(ValueError, match="up|down"):
        load_roadmap(directory)
    (directory / "a.md").write_text(
        GOAL_A.replace("up: [b]", "up: []")
        .replace("down: []", "down: []")
        .replace("prereqs: []", "prereqs: [b]")
    )
    with pytest.raises(ValueError, match="graph contains a cycle"):
        load_roadmap(directory)


def test_goal_document_is_frozen() -> None:
    goal = parse_goal_document(GOAL_A, slug="a")
    with pytest.raises(AttributeError, match="immutable"):
        goal.title = "Changed"  # type: ignore[misc]


def test_edges_union_prereqs_first_and_preserves_order() -> None:
    text = (
        GOAL_A.replace("up: [b]", "up: []")
        .replace("prereqs: []", "prereqs: [p, shared]")
        .replace("down: []", "down: [shared, child]")
    )
    goal = parse_goal_document(text, slug="a")
    assert goal.edges == ("p", "shared", "child")


@pytest.mark.parametrize(
    ("field", "value"),
    [("kind", "roadmap"), ("slug", "other"), ("roadmap", "../bad")],
)
def test_invalid_goal_metadata_names_file_and_field(field: str, value: str) -> None:
    defaults = {"kind": "goal", "slug": "a", "roadmap": "demo-roadmap"}
    text = GOAL_A.replace(f"{field}: {defaults[field]}", f"{field}: {value}")
    with pytest.raises(ValueError, match=rf"a\.md.*{field}"):
        parse_goal_document(text, slug="a")


def test_unresolved_up_names_file_and_field(tmp_path: Path) -> None:
    directory = tmp_path / "demo-roadmap"
    directory.mkdir()
    (directory / "index.md").write_text(ROADMAP)
    (directory / "a.md").write_text(GOAL_A.replace("up: [b]", "up: [missing]"))
    with pytest.raises(ValueError, match=r"a\.md.*up.*missing"):
        load_roadmap(directory)


def test_down_cycle_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "demo-roadmap"
    directory.mkdir()
    (directory / "index.md").write_text(ROADMAP)
    (directory / "a.md").write_text(GOAL_A.replace("down: []", "down: [b]"))
    (directory / "b.md").write_text(
        GOAL_B.replace("prereqs: [a, a]", "prereqs: []")
        .replace("up: []", "up: [a]")
        .replace("down: []", "down: [a]")
    )
    with pytest.raises(ValueError, match=r"graph contains a cycle"):
        load_roadmap(directory)


@pytest.mark.parametrize(
    "body",
    [
        "# A\n\n# Duplicate\n\n## Intent\nDo A.\n\n## Acceptance\nA works.\n",
        "~~~\n# Fenced\n~~~\n\n# A\n\n## Intent\nDo A.\n\n## Acceptance\nA works.\n",
        "# A\n\n## Notes\nNope.\n\n## Intent\nDo A.\n\n## Acceptance\nA works.\n",
        "# A\n\n## Intent\nDo A.\n\n## Intent\nAgain.\n\n## Acceptance\nA works.\n",
    ],
)
def test_heading_noise_is_ignored(body: str) -> None:
    text = GOAL_A.split("# Goal A", 1)[0] + body
    goal = parse_goal_document(text, slug="a")
    assert goal.title == "A"
    assert goal.intent == "Do A."
    assert goal.acceptance == "A works."


def test_missing_h1_still_fails() -> None:
    text = GOAL_A.split("# Goal A", 1)[0] + "## Intent\nDo A.\n\n## Acceptance\nA works.\n"
    with pytest.raises(ValueError, match=r"a\.md.*missing H1"):
        parse_goal_document(text, slug="a")
