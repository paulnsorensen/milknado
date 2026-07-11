"""MCP veneer for the GitHub Projects crossover: import + bind + export tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.adapters.gh import GhTransportError
from milknado.domains.github import bind as bind_mod
from milknado.domains.github import exporter as exp_mod
from milknado.domains.github import importer as imp_mod
from milknado.mcp_github import (
    milknado_github_roadmap_bind,
    milknado_github_roadmap_export,
    milknado_github_roadmap_import,
)

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

GOAL_MD = """---
kind: goal
slug: g1
roadmap: demo
created: 2026-06-02
status: pending
prereqs: []
---
# Goal one

## Intent
do the thing
"""


def _call(tool, **kwargs):  # noqa: ANN001, ANN003
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


def _seed_wiki(project_root: Path) -> None:
    d = project_root / ".hallouminate" / "wiki" / "roadmaps" / "demo"
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "g1.md").write_text(GOAL_MD)


def _stub_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(imp_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(imp_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_1", "title": "RM"})
    monkeypatch.setattr(imp_mod, "gh_item_list", lambda _o, _n: [{"id": "PVTI_1", "title": "g"}])


def test_import_tool_seeds_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_import(monkeypatch)
    result = _call(
        milknado_github_roadmap_import, owner="acme", number=7, project_root=str(tmp_path)
    )
    assert result["created"] == 2
    assert result["goal_node_ids"] == {"PVTI_1": result["goal_node_ids"]["PVTI_1"]}


def test_import_tool_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> None:
        raise GhTransportError("gh auth refresh -s project")

    monkeypatch.setattr(imp_mod, "gh_preflight", _boom)
    with pytest.raises(GhTransportError):
        _call(milknado_github_roadmap_import, owner="acme", number=7, project_root=str(tmp_path))


def test_bind_tool_projects_roadmap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from milknado._mcp_core import open_graph, resolve_project_root
    from milknado.domains.wiki.importer import import_roadmap

    _seed_wiki(tmp_path)
    root = resolve_project_root(str(tmp_path))
    graph, _cfg = open_graph(root)
    import_roadmap(root / ".hallouminate" / "wiki", "demo", graph)
    graph.close()

    monkeypatch.setattr(bind_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(bind_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_1"})
    monkeypatch.setattr(bind_mod, "gh_issue_create", lambda *_a: "https://x/1")
    monkeypatch.setattr(bind_mod, "gh_item_add", lambda _o, _n, _u: "PVTI_1")
    monkeypatch.setattr(bind_mod, "gh_field_list", lambda _o, _n: [])
    monkeypatch.setattr(bind_mod, "gh_field_create", lambda *_a: {"id": "F"})
    monkeypatch.setattr(bind_mod, "gh_field_create_text", lambda *_a: {"id": "F2"})

    result = _call(milknado_github_roadmap_bind, roadmap_slug="demo", project_root=str(tmp_path))
    assert result["issues_created"] == 1
    assert result["field_created"] is True


def test_export_tool_after_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_import(monkeypatch)
    _call(milknado_github_roadmap_import, owner="acme", number=7, project_root=str(tmp_path))

    monkeypatch.setattr(exp_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(exp_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_1"})
    monkeypatch.setattr(
        exp_mod,
        "gh_field_list",
        lambda _o, _n: [
            {"id": "F_s", "name": "Milknado Status", "options": [{"id": "o", "name": "Pending"}]},
            {"id": "F_h", "name": "Milknado Harvest"},
        ],
    )
    monkeypatch.setattr(
        exp_mod, "gh_item_list", lambda _o, _n: [{"id": "PVTI_1", "url": "https://x/1"}]
    )
    monkeypatch.setattr(exp_mod, "gh_item_edit", lambda *_a, **_k: None)
    monkeypatch.setattr(exp_mod, "gh_issue_edit_body", lambda *_a: None)

    result = _call(
        milknado_github_roadmap_export, owner="acme", number=7, project_root=str(tmp_path)
    )
    assert result["goals_exported"] == 1
    assert result["bodies_overwritten"] == 0  # github-origin: body never touched


def test_export_tool_before_import_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(exp_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_unbound"})
    with pytest.raises(LookupError):
        _call(milknado_github_roadmap_export, owner="acme", number=7, project_root=str(tmp_path))
