"""`bind_github_project`: wiki-origin projection (Acceptance 3, 5).

Creates one Issue per goal (body = wiki Intent), adds items, stores github_ref on
roadmap + goals, and bootstraps the Status field create-once — never an
add-option-to-existing-field call. gh transport is monkeypatched at the bind
module's own `gh_*` names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.github import bind as bind_mod
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
    """Records the bind's gh calls; returns a distinct item id per issue."""

    def __init__(self, fields: list[dict] | None = None) -> None:
        self.fields = fields or []
        self.issues: list[tuple[str, str, str, str]] = []
        self.items: list[str] = []
        self.created_fields: list[tuple[str, list[str]]] = []
        self.created_text_fields: list[str] = []
        self._counter = 0

    def issue_create(self, owner: str, repo: str, title: str, body: str) -> str:
        self.issues.append((owner, repo, title, body))
        self._counter += 1
        return f"https://github.com/{owner}/{repo}/issues/{self._counter}"

    def item_add(self, _owner: str, _number: int, url: str) -> str:
        self.items.append(url)
        return f"PVTI_{len(self.items)}"

    def field_list(self, _owner: str, _number: int) -> list[dict]:
        return list(self.fields)

    def field_create(self, _number: int, _owner: str, name: str, options: list[str]) -> dict:
        self.created_fields.append((name, options))
        return {"id": "F_status"}

    def field_create_text(self, _number: int, _owner: str, name: str) -> dict:
        self.created_text_fields.append(name)
        return {"id": "F_harvest"}


def _wire(monkeypatch: pytest.MonkeyPatch, fake: FakeGh) -> None:
    monkeypatch.setattr(bind_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(bind_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_1"})
    monkeypatch.setattr(bind_mod, "gh_issue_create", fake.issue_create)
    monkeypatch.setattr(bind_mod, "gh_item_add", fake.item_add)
    monkeypatch.setattr(bind_mod, "gh_field_list", fake.field_list)
    monkeypatch.setattr(bind_mod, "gh_field_create", fake.field_create)
    monkeypatch.setattr(bind_mod, "gh_field_create_text", fake.field_create_text)


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


def test_bind_creates_issue_and_item_per_goal(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, wiki_root = _import(tmp_path, graph)
    fake = FakeGh()
    _wire(monkeypatch, fake)
    result = bind_github_project(graph, rid, wiki_root)
    assert result.issues_created == 2
    assert result.items_added == 2
    assert len(fake.issues) == 2
    # Issue bodies mirror the wiki Intent.
    bodies = {body for *_h, body in fake.issues}
    assert "build the first goal" in bodies
    assert "build the second goal" in bodies


def test_bind_stores_refs_on_roadmap_and_goals(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, wiki_root = _import(tmp_path, graph)
    _wire(monkeypatch, FakeGh())
    bind_github_project(graph, rid, wiki_root)
    assert graph.get_node(rid).github_ref == "PVT_1"
    goals = graph.get_children(rid)
    assert all(g.github_ref is not None for g in goals)


def test_bind_creates_status_field_once_with_all_options(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, wiki_root = _import(tmp_path, graph)
    fake = FakeGh(fields=[])  # no pre-existing fields
    _wire(monkeypatch, fake)
    result = bind_github_project(graph, rid, wiki_root)
    assert result.field_created is True
    assert fake.created_fields == [(STATUS_FIELD_NAME, STATUS_OPTIONS)]
    assert len(fake.created_text_fields) == 1  # harvest text field too


def test_bind_skips_field_create_when_present(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, wiki_root = _import(tmp_path, graph)
    fake = FakeGh(
        fields=[
            {"id": "F_status", "name": "Milknado Status"},
            {"id": "F_harvest", "name": "Milknado Harvest"},
        ]
    )
    _wire(monkeypatch, fake)
    result = bind_github_project(graph, rid, wiki_root)
    assert result.field_created is False
    assert fake.created_fields == []  # never an add-option / re-create path
    assert fake.created_text_fields == []


def test_bind_skips_already_bound_goals(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, wiki_root = _import(tmp_path, graph)
    already = graph.get_children(rid)[0]
    graph.set_github_ref(already.id, "PVTI_pre")
    fake = FakeGh()
    _wire(monkeypatch, fake)
    result = bind_github_project(graph, rid, wiki_root)
    assert result.issues_created == 1  # only the still-unbound goal


def test_bind_accepts_explicit_owner_number(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Explicit args bypass the github_project frontmatter parse.
    index = INDEX_MD.replace("github_project: acme/7\n", "")
    rid, wiki_root = _import(tmp_path, graph, index=index)
    _wire(monkeypatch, FakeGh())
    result = bind_github_project(graph, rid, wiki_root, owner="acme", number=7)
    assert result.issues_created == 2


class TestBindGuards:
    def test_non_roadmap_node_raises(self, graph: MikadoGraph, tmp_path: Path) -> None:
        node = graph.add_node("task-ish")
        with pytest.raises(ValueError, match="roadmap"):
            bind_github_project(graph, node.id, tmp_path)

    def test_roadmap_without_wiki_ref_raises(self, graph: MikadoGraph, tmp_path: Path) -> None:
        node = graph.add_node("rm", spec=NodeSpec(kind=NodeKind.ROADMAP))
        with pytest.raises(ValueError, match="wiki_ref"):
            bind_github_project(graph, node.id, tmp_path)

    def test_missing_github_repo_frontmatter_raises(
        self, tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = INDEX_MD.replace("github_repo: acme/repo\n", "")
        rid, wiki_root = _import(tmp_path, graph, index=index)
        _wire(monkeypatch, FakeGh())
        with pytest.raises(ValueError, match="github_repo"):
            bind_github_project(graph, rid, wiki_root)

    def test_missing_github_project_frontmatter_raises(
        self, tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = INDEX_MD.replace("github_project: acme/7\n", "")
        rid, wiki_root = _import(tmp_path, graph, index=index)
        _wire(monkeypatch, FakeGh())
        with pytest.raises(ValueError, match="github_project"):
            bind_github_project(graph, rid, wiki_root)

    def test_malformed_github_project_raises(
        self, tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = INDEX_MD.replace("github_project: acme/7", "github_project: acme/notanumber")
        rid, wiki_root = _import(tmp_path, graph, index=index)
        _wire(monkeypatch, FakeGh())
        with pytest.raises(ValueError, match="malformed `github_project`"):
            bind_github_project(graph, rid, wiki_root)

    def test_malformed_github_repo_raises(
        self, tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = INDEX_MD.replace("github_repo: acme/repo", "github_repo: acme/")
        rid, wiki_root = _import(tmp_path, graph, index=index)
        _wire(monkeypatch, FakeGh())
        with pytest.raises(ValueError, match="malformed `github_repo`"):
            bind_github_project(graph, rid, wiki_root)


class TestGoalIntent:
    def test_intent_section_read_from_file(self, tmp_path: Path, graph: MikadoGraph) -> None:
        rid, wiki_root = _import(tmp_path, graph)
        roadmap = graph.get_node(rid)
        file_map = goal_file_map(wiki_root, roadmap.wiki_ref)
        goal = graph.get_children(rid)[0]
        assert "build the" in goal_intent(goal, file_map)

    def test_intent_falls_back_to_description_without_wiki_ref(self, graph: MikadoGraph) -> None:
        goal = graph.add_node("bare goal", spec=NodeSpec(kind=NodeKind.GOAL))
        assert goal_intent(goal, {}) == "bare goal"
