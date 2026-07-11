"""`export_github_roadmap`: the milknado-owned half of the GitHub membrane
(Acceptance 2, 4).

Always writes Status + harvest per bound goal; overwrites the Issue body ONLY for
wiki-origin goals (wiki_ref present) — a github-origin goal's body is never
touched. gh transport is monkeypatched at the exporter module's own `gh_*` names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.github import exporter as exp_mod
from milknado.domains.github.exporter import (
    export_github_roadmap,
    resolve_github_roadmap_node,
)
from milknado.domains.graph import MikadoGraph
from milknado.domains.wiki.importer import import_roadmap

SLUG = "demo-roadmap"

INDEX_MD = """---
kind: roadmap
slug: demo-roadmap
created: 2026-06-01
status: pending
---
# Demo Roadmap
prose
"""

GOAL_MD = """---
kind: goal
slug: wire-export
roadmap: demo-roadmap
created: 2026-06-03
status: pending
prereqs: []
---
# Wire the export

## Intent
Export without clobbering human text.

## Acceptance
- done
"""


def _status_field(*, with_options: bool = True) -> dict:
    options = (
        [
            {"id": "opt-pending", "name": "Pending"},
            {"id": "opt-running", "name": "Running"},
            {"id": "opt-done", "name": "Done"},
            {"id": "opt-failed", "name": "Failed"},
        ]
        if with_options
        else []
    )
    return {"id": "F_status", "name": "Milknado Status", "options": options}


HARVEST_FIELD = {"id": "F_harvest", "name": "Milknado Harvest"}


class FakeExport:
    def __init__(self, fields: list[dict], items: list[dict]) -> None:
        self.fields = fields
        self.items = items
        self.item_edits: list[dict] = []
        self.body_edits: list[tuple[str, str]] = []

    def field_list(self, _o: str, _n: int) -> list[dict]:
        return self.fields

    def item_list(self, _o: str, _n: int) -> list[dict]:
        return self.items

    def item_edit(self, project_id, item_id, field_id, *, text=None, single_select_option_id=None):  # noqa: ANN001
        self.item_edits.append(
            {
                "item_id": item_id,
                "field_id": field_id,
                "text": text,
                "option": single_select_option_id,
            }
        )

    def issue_edit_body(self, url: str, body: str) -> None:
        self.body_edits.append((url, body))


def _wire(monkeypatch: pytest.MonkeyPatch, fake: FakeExport) -> None:
    monkeypatch.setattr(exp_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(exp_mod, "gh_field_list", fake.field_list)
    monkeypatch.setattr(exp_mod, "gh_item_list", fake.item_list)
    monkeypatch.setattr(exp_mod, "gh_item_edit", fake.item_edit)
    monkeypatch.setattr(exp_mod, "gh_issue_edit_body", fake.issue_edit_body)


def _seed(tmp_path: Path, graph: MikadoGraph) -> tuple[int, int, int, Path]:
    """Seed a wiki roadmap + one wiki-origin goal and one github-origin goal.

    Returns (roadmap_id, wiki_goal_id, github_goal_id, wiki_root).
    """
    d = tmp_path / ".hallouminate" / "wiki" / "roadmaps" / SLUG
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "wire-export.md").write_text(GOAL_MD)
    wiki_root = tmp_path / ".hallouminate" / "wiki"
    result = import_roadmap(wiki_root, SLUG, graph)
    rid = result.roadmap_node_id
    wiki_goal_id = result.goal_node_ids["wire-export"]
    graph.set_github_ref(rid, "PVT_1")
    graph.set_github_ref(wiki_goal_id, "PVTI_wiki")
    gh_goal = graph.add_node(
        "github goal", parent_id=rid, spec=NodeSpec(kind=NodeKind.GOAL, github_ref="PVTI_gh")
    )
    return rid, wiki_goal_id, gh_goal.id, wiki_root


def _full_fake() -> FakeExport:
    return FakeExport(
        fields=[_status_field(), HARVEST_FIELD],
        items=[
            {"id": "PVTI_wiki", "url": "https://x/1"},
            {"id": "PVTI_gh", "url": "https://x/2"},
        ],
    )


def test_wiki_origin_overwrites_body_github_origin_does_not(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, wiki_goal_id, _gh, wiki_root = _seed(tmp_path, graph)
    graph.mark_running(wiki_goal_id)
    graph.mark_done(wiki_goal_id)
    fake = _full_fake()
    _wire(monkeypatch, fake)
    result = export_github_roadmap(graph, rid, wiki_root, owner="acme", number=7)
    assert result.goals_exported == 2
    assert result.bodies_overwritten == 1
    # Only the wiki-origin item's Issue body was touched.
    assert [url for url, _b in fake.body_edits] == ["https://x/1"]
    assert "Export without clobbering" in fake.body_edits[0][1]


def test_status_option_mapping_written(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, wiki_goal_id, _gh, wiki_root = _seed(tmp_path, graph)
    graph.mark_running(wiki_goal_id)
    graph.mark_done(wiki_goal_id)
    fake = _full_fake()
    _wire(monkeypatch, fake)
    export_github_roadmap(graph, rid, wiki_root, owner="acme", number=7)
    option_edits = [e for e in fake.item_edits if e["item_id"] == "PVTI_wiki" and e["option"]]
    assert option_edits[0]["option"] == "opt-done"


def test_harvest_text_written_per_goal(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, _wg, _gh, wiki_root = _seed(tmp_path, graph)
    fake = _full_fake()
    _wire(monkeypatch, fake)
    export_github_roadmap(graph, rid, wiki_root, owner="acme", number=7)
    text_edits = [e for e in fake.item_edits if e["field_id"] == "F_harvest"]
    assert {e["item_id"] for e in text_edits} == {"PVTI_wiki", "PVTI_gh"}
    assert all(e["text"] is not None for e in text_edits)


def test_blocked_status_skips_option_but_writes_harvest(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, _wg, gh_goal_id, wiki_root = _seed(tmp_path, graph)
    graph._conn.execute("UPDATE nodes SET status = 'blocked' WHERE id = ?", (gh_goal_id,))
    graph._conn.commit()
    fake = _full_fake()
    _wire(monkeypatch, fake)
    export_github_roadmap(graph, rid, wiki_root, owner="acme", number=7)
    gh_option_edits = [
        e for e in fake.item_edits if e["item_id"] == "PVTI_gh" and e["option"] is not None
    ]
    gh_text_edits = [
        e for e in fake.item_edits if e["item_id"] == "PVTI_gh" and e["text"] is not None
    ]
    assert gh_option_edits == []  # blocked has no Status option
    assert len(gh_text_edits) == 1  # harvest still written


def test_missing_option_id_skips_status_write(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid, wiki_goal_id, _gh, wiki_root = _seed(tmp_path, graph)
    graph.mark_running(wiki_goal_id)
    graph.mark_done(wiki_goal_id)
    fake = FakeExport(
        fields=[_status_field(with_options=False), HARVEST_FIELD],
        items=[{"id": "PVTI_wiki", "url": "https://x/1"}, {"id": "PVTI_gh", "url": "https://x/2"}],
    )
    _wire(monkeypatch, fake)
    export_github_roadmap(graph, rid, wiki_root, owner="acme", number=7)
    assert [e for e in fake.item_edits if e["option"] is not None] == []
    # harvest text still written for both goals
    assert len([e for e in fake.item_edits if e["text"] is not None]) == 2


def test_wiki_origin_body_skipped_when_item_absent_from_project(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Acceptance 4 guard: a wiki-origin goal whose item was removed from the
    # Project (no url in item-list) still gets Status + harvest written by id,
    # but the body mirror is skipped — never a body edit against a missing url.
    rid, wiki_goal_id, _gh, wiki_root = _seed(tmp_path, graph)
    graph.mark_running(wiki_goal_id)
    graph.mark_done(wiki_goal_id)
    fake = FakeExport(
        fields=[_status_field(), HARVEST_FIELD],
        items=[{"id": "PVTI_gh", "url": "https://x/2"}],  # PVTI_wiki dropped
    )
    _wire(monkeypatch, fake)
    result = export_github_roadmap(graph, rid, wiki_root, owner="acme", number=7)
    assert result.bodies_overwritten == 0
    assert fake.body_edits == []
    # Status + harvest still written for the wiki goal by its item id.
    wiki_edits = [e for e in fake.item_edits if e["item_id"] == "PVTI_wiki"]
    assert any(e["text"] is not None for e in wiki_edits)


def test_malformed_item_without_id_is_skipped(
    tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    # gh item-list may return a draft/malformed entry with no `id`; building the
    # url map must skip it deterministically rather than KeyError the whole export.
    rid, wiki_goal_id, _gh, wiki_root = _seed(tmp_path, graph)
    graph.mark_running(wiki_goal_id)
    graph.mark_done(wiki_goal_id)
    fake = FakeExport(
        fields=[_status_field(), HARVEST_FIELD],
        items=[
            {"url": "https://x/0"},  # malformed: no id
            {"id": "PVTI_wiki", "url": "https://x/1"},
            {"id": "PVTI_gh", "url": "https://x/2"},
        ],
    )
    _wire(monkeypatch, fake)
    result = export_github_roadmap(graph, rid, wiki_root, owner="acme", number=7)
    assert result.goals_exported == 2
    assert result.bodies_overwritten == 1


class TestExportGuards:
    def test_non_roadmap_node_raises(self, graph: MikadoGraph, tmp_path: Path) -> None:
        node = graph.add_node("plain")
        with pytest.raises(ValueError, match="roadmap"):
            export_github_roadmap(graph, node.id, tmp_path, owner="a", number=1)

    def test_roadmap_without_github_ref_raises(self, graph: MikadoGraph, tmp_path: Path) -> None:
        node = graph.add_node("rm", spec=NodeSpec(kind=NodeKind.ROADMAP))
        with pytest.raises(ValueError, match="github_ref"):
            export_github_roadmap(graph, node.id, tmp_path, owner="a", number=1)

    def test_missing_fields_raises(
        self, tmp_path: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rid, _wg, _gh, wiki_root = _seed(tmp_path, graph)
        _wire(monkeypatch, FakeExport(fields=[], items=[]))
        with pytest.raises(ValueError, match="missing"):
            export_github_roadmap(graph, rid, wiki_root, owner="acme", number=7)


class TestResolveNode:
    def test_resolve_finds_bound_roadmap(
        self, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node = graph.add_node("rm", spec=NodeSpec(kind=NodeKind.ROADMAP, github_ref="PVT_9"))
        monkeypatch.setattr(exp_mod, "gh_preflight", lambda: None)
        monkeypatch.setattr(exp_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_9"})
        found = resolve_github_roadmap_node(graph, "acme", 9)
        assert found.id == node.id

    def test_resolve_unbound_project_raises(
        self, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(exp_mod, "gh_preflight", lambda: None)
        monkeypatch.setattr(exp_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_absent"})
        with pytest.raises(LookupError, match="not bound"):
            resolve_github_roadmap_node(graph, "acme", 9)
