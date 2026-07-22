"""`bind_github_project`: wiki-origin projection through an injected port."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.github._fields import STATUS_FIELD_NAME, STATUS_OPTIONS
from milknado.domains.github._intent import goal_file_map, goal_intent
from milknado.domains.github.bind import bind_github_project
from milknado.domains.graph import MikadoGraph
from milknado.domains.wiki.importer import import_roadmap

SLUG = "demo"

INDEX_MD = """---
kind: roadmap
slug: demo
created: 2026-06-01
status: pending
github_project: acme/7
github_repo: acme/repo
---
# Demo
prose
"""

GOAL_A = """---
kind: goal
slug: g1
roadmap: demo
created: 2026-06-02
status: pending
prereqs: []
---
# Goal one

## Intent
build the first goal

## Acceptance
- done
"""

GOAL_B = """---
kind: goal
slug: g2
roadmap: demo
created: 2026-06-03
status: pending
prereqs: []
---
# Goal two

## Intent
build the second goal
"""


class FakeGh:
    """Records bind operations and exposes the injected GitHub port."""

    def __init__(
        self,
        fields: list[dict] | None = None,
        existing_items: list[dict] | None = None,
        *,
        fail_item_add_once: bool = False,
    ) -> None:
        self.fields = fields or []
        self.existing_items = existing_items or []
        self.fail_item_add_once = fail_item_add_once
        self.issues: list[tuple[str, str, str, str]] = []
        self.items: list[str] = []
        self.created_fields: list[tuple[str, list[str]]] = []
        self.created_text_fields: list[str] = []
        self._counter = 0

    def preflight(self) -> None:
        pass

    def project_view(self, _owner: str, _number: int) -> dict:
        return {"id": "PVT_1"}

    def issue_create(self, owner: str, repo: str, title: str, body: str) -> str:
        self.issues.append((owner, repo, title, body))
        self._counter += 1
        return f"https://github.com/{owner}/{repo}/issues/{self._counter}"

    def item_add(self, _owner: str, _number: int, url: str) -> str:
        if self.fail_item_add_once:
            self.fail_item_add_once = False
            raise RuntimeError("simulated item_add failure")
        self.items.append(url)
        return f"PVTI_{len(self.items)}"

    def item_list(self, _owner: str, _number: int) -> list[dict]:
        return list(self.existing_items)

    def field_list(self, _owner: str, _number: int) -> list[dict]:
        return list(self.fields)

    def field_create(self, _number: int, _owner: str, name: str, options: list[str]) -> dict:
        self.created_fields.append((name, options))
        return {"id": "F_status"}

    def field_create_text(self, _number: int, _owner: str, name: str) -> dict:
        self.created_text_fields.append(name)
        return {"id": "F_harvest"}


def _seed_wiki(tmp_path: Path, *, index: str = INDEX_MD) -> Path:
    d = tmp_path / ".hallouminate" / "wiki" / "roadmaps" / SLUG
    d.mkdir(parents=True)
    (d / "index.md").write_text(index)
    (d / "g1.md").write_text(GOAL_A)
    (d / "g2.md").write_text(GOAL_B)
    return tmp_path / ".hallouminate" / "wiki"


def _import(tmp_path: Path, graph: MikadoGraph, **kw: str) -> tuple[int, Path]:
    wiki_root = _seed_wiki(tmp_path, **kw)
    return import_roadmap(wiki_root, SLUG, graph).roadmap_node_id, wiki_root


def test_bind_creates_issue_and_item_per_goal(tmp_path: Path, graph: MikadoGraph) -> None:
    rid, root = _import(tmp_path, graph)
    github = FakeGh()

    result = bind_github_project(graph, rid, root, github)
    assert result.issues_created == 2
    assert result.items_added == 2
    bodies = [body for *_head, body in github.issues]
    assert [body.splitlines()[0] for body in bodies] == [
        "build the first goal",
        "build the second goal",
    ]
    assert all("milknado-bind:roadmap=" in body for body in bodies)
    assert graph.get_node(rid).github_ref == "PVT_1"
    assert all(goal.github_ref for goal in graph.get_children(rid))


def test_bind_creates_missing_fields_once(tmp_path: Path, graph: MikadoGraph) -> None:
    rid, root = _import(tmp_path, graph)
    github = FakeGh()

    result = bind_github_project(graph, rid, root, github)

    assert result.field_created is True
    assert github.created_fields == [(STATUS_FIELD_NAME, STATUS_OPTIONS)]
    assert github.created_text_fields == ["Milknado Harvest"]


def test_bind_preserves_existing_fields(tmp_path: Path, graph: MikadoGraph) -> None:
    rid, root = _import(tmp_path, graph)
    github = FakeGh(
        fields=[
            {"id": "F_status", "name": "Milknado Status"},
            {"id": "F_harvest", "name": "Milknado Harvest"},
        ]
    )

    result = bind_github_project(graph, rid, root, github)

    assert result.field_created is False
    assert github.created_fields == []
    assert github.created_text_fields == []


def test_bind_skips_already_bound_goals(tmp_path: Path, graph: MikadoGraph) -> None:
    rid, root = _import(tmp_path, graph)
    already = graph.get_children(rid)[0]
    graph.set_github_ref(already.id, "PVTI_pre")

    result = bind_github_project(graph, rid, root, FakeGh())

    assert result.issues_created == 1
    assert graph.get_node(already.id).github_ref == "PVTI_pre"


def test_bind_accepts_explicit_owner_number(tmp_path: Path, graph: MikadoGraph) -> None:
    index = INDEX_MD.replace("github_project: acme/7\n", "")
    rid, root = _import(tmp_path, graph, index=index)

    result = bind_github_project(graph, rid, root, FakeGh(), owner="acme", number=7)

    assert result.issues_created == 2


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("github_repo: acme/repo\n", "", "github_repo"),
        ("github_project: acme/7\n", "", "github_project"),
        ("github_project: acme/7", "github_project: acme/notanumber", "malformed"),
        ("github_repo: acme/repo", "github_repo: acme/", "malformed"),
    ],
)
def test_bind_rejects_invalid_frontmatter(
    tmp_path: Path, graph: MikadoGraph, old: str, new: str, message: str
) -> None:
    rid, root = _import(tmp_path, graph, index=INDEX_MD.replace(old, new))

    with pytest.raises(ValueError, match=message):
        bind_github_project(graph, rid, root, FakeGh())


def test_bind_rejects_invalid_node(graph: MikadoGraph, tmp_path: Path) -> None:
    node = graph.add_node("task-ish")
    with pytest.raises(ValueError, match="roadmap"):
        bind_github_project(graph, node.id, tmp_path, FakeGh())


def test_bind_rejects_roadmap_without_wiki_ref(graph: MikadoGraph, tmp_path: Path) -> None:
    node = graph.add_node("rm", spec=NodeSpec(kind=NodeKind.ROADMAP))
    with pytest.raises(ValueError, match="wiki_ref"):
        bind_github_project(graph, node.id, tmp_path, FakeGh())


def test_bind_rejects_different_project(tmp_path: Path, graph: MikadoGraph) -> None:
    rid, root = _import(tmp_path, graph)
    graph.set_github_ref(rid, "PVT_other")

    with pytest.raises(ValueError, match="PVT_other.*PVT_1"):
        bind_github_project(graph, rid, root, FakeGh())


def test_same_project_binding_is_idempotent(tmp_path: Path, graph: MikadoGraph) -> None:
    rid, root = _import(tmp_path, graph)
    graph.set_github_ref(rid, "PVT_1")

    bind_github_project(graph, rid, root, FakeGh())

    assert graph.get_node(rid).github_ref == "PVT_1"


def test_same_title_unrelated_item_is_never_adopted(tmp_path: Path, graph: MikadoGraph) -> None:
    rid, root = _import(tmp_path, graph)
    target = next(goal for goal in graph.get_children(rid) if goal.description == "Goal one")
    github = FakeGh(
        existing_items=[
            {
                "id": "PVTI_unrelated",
                "title": "Goal one",
                "url": "https://github.com/other/repo/issues/99",
            }
        ]
    )

    result = bind_github_project(graph, rid, root, github)

    assert graph.get_node(target.id).github_ref != "PVTI_unrelated"
    assert {title for *_prefix, title, _body in github.issues} == {"Goal one", "Goal two"}
    assert result.issues_created == 2
    assert result.items_added == 2


def test_bind_recovers_item_add_after_issue_creation(tmp_path: Path, graph: MikadoGraph) -> None:
    rid, root = _import(tmp_path, graph)
    github = FakeGh(fail_item_add_once=True)

    with pytest.raises(RuntimeError, match="simulated item_add failure"):
        bind_github_project(graph, rid, root, github)

    first_goal = next(goal for goal in graph.get_children(rid) if goal.description == "Goal one")
    attempt = graph.get_github_bind_attempt(first_goal.id)
    assert attempt is not None
    assert attempt["issue_url"].endswith("/issues/1")
    assert first_goal.github_ref is None
    assert len(github.issues) == 1

    result = bind_github_project(graph, rid, root, github)

    assert result.issues_created == 1
    assert result.items_added == 2
    assert len(github.issues) == 2
    assert graph.get_github_bind_attempt(first_goal.id) is None
    assert graph.get_node(first_goal.id).github_ref == "PVTI_1"


def test_bind_recovers_pending_attempt_from_existing_marker(
    tmp_path: Path, graph: MikadoGraph
) -> None:
    from milknado.domains.github.bind import _correlation_marker

    rid, root = _import(tmp_path, graph)
    goal = graph.get_children(rid)[0]
    roadmap = graph.get_node(rid)
    marker = _correlation_marker(roadmap.wiki_ref, goal.wiki_ref or str(goal.id))
    graph.set_github_bind_attempt(goal.id, marker, "https://example/issues/9", "now")
    github = FakeGh(existing_items=[{"id": "PVTI_recovered", "body": marker}])

    result = bind_github_project(graph, rid, root, github)

    assert graph.get_node(goal.id).github_ref == "PVTI_recovered"
    assert result.issues_created == 1


def test_bind_rejects_unknown_issue_creation_outcome(tmp_path: Path, graph: MikadoGraph) -> None:
    from milknado.domains.github.bind import _correlation_marker

    rid, root = _import(tmp_path, graph)
    goal = graph.get_children(rid)[0]
    roadmap = graph.get_node(rid)
    marker = _correlation_marker(roadmap.wiki_ref, goal.wiki_ref or str(goal.id))
    graph.set_github_bind_attempt(goal.id, marker, None, "now")

    with pytest.raises(RuntimeError, match="outcome is unknown"):
        bind_github_project(graph, rid, root, FakeGh())


@pytest.mark.parametrize("with_attempt", [False, True])
def test_bind_rejects_ambiguous_marker_matches(
    tmp_path: Path, graph: MikadoGraph, with_attempt: bool
) -> None:
    from milknado.domains.github.bind import _correlation_marker

    rid, root = _import(tmp_path, graph)
    goal = graph.get_children(rid)[0]
    roadmap = graph.get_node(rid)
    marker = _correlation_marker(roadmap.wiki_ref, goal.wiki_ref or str(goal.id))
    if with_attempt:
        graph.set_github_bind_attempt(goal.id, marker, "https://example/issues/9", "now")
    github = FakeGh(
        existing_items=[
            {"id": "PVTI_a", "body": marker},
            {"id": "PVTI_b", "body": marker},
        ]
    )

    with pytest.raises(RuntimeError, match="matched 2 project items"):
        bind_github_project(graph, rid, root, github)


def test_bind_recovers_existing_marker_without_pending_attempt(
    tmp_path: Path, graph: MikadoGraph
) -> None:
    from milknado.domains.github.bind import _correlation_marker

    rid, root = _import(tmp_path, graph)
    roadmap = graph.get_node(rid)
    goal = graph.get_children(rid)[0]
    marker = _correlation_marker(roadmap.wiki_ref, goal.wiki_ref or str(goal.id))
    github = FakeGh(existing_items=[{"id": "PVTI_existing", "body": marker}])

    bind_github_project(graph, rid, root, github)

    assert graph.get_node(goal.id).github_ref == "PVTI_existing"


class TestGoalIntent:
    def test_intent_section_read_from_file(self, tmp_path: Path, graph: MikadoGraph) -> None:
        rid, wiki_root = _import(tmp_path, graph)
        roadmap = graph.get_node(rid)
        file_map = goal_file_map(wiki_root, roadmap.wiki_ref)
        goal = graph.get_children(rid)[0]
        assert "build the" in goal_intent(goal, file_map, wiki_root)

    def test_intent_falls_back_to_description_without_wiki_ref(self, graph: MikadoGraph) -> None:
        goal = graph.add_node("bare goal", spec=NodeSpec(kind=NodeKind.GOAL))
        assert goal_intent(goal, {}, Path()) == "bare goal"
