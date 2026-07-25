"""MCP veneer for the GitHub Projects crossover: import + bind + export tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from milknado.adapters.gh import GhProjectAdapter, GhTransportError
from milknado.domains.github import GithubField, GithubItem, GithubProject
from milknado.mcp.github import (
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


def _call(tool: Any, **kwargs: Any) -> Any:
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


def _stub_adapter(monkeypatch: pytest.MonkeyPatch, name: str, value: Any) -> None:
    monkeypatch.setattr(GhProjectAdapter, name, staticmethod(value))


def _seed_wiki(project_root: Path) -> None:
    d = project_root / ".hallouminate" / "wiki" / "roadmaps" / "demo"
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "g1.md").write_text(GOAL_MD)


def _stub_import(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_adapter(monkeypatch, "preflight", lambda: None)
    _stub_adapter(
        monkeypatch, "project_view", lambda _o, _n: GithubProject(id="PVT_1", title="RM")
    )
    _stub_adapter(monkeypatch, "item_list", lambda _o, _n: [GithubItem(id="PVTI_1", title="g")])


def test_import_tool_seeds_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_import(monkeypatch)
    result = _call(
        milknado_github_roadmap_import, owner="acme", number=7, project_root=str(tmp_path)
    )
    assert result["created"] == 2
    assert set(result["goal_node_ids"]) == {"PVTI_1"}
    assert isinstance(result["goal_node_ids"]["PVTI_1"], int)


def test_import_tool_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> None:
        raise GhTransportError("gh auth refresh -s project")

    _stub_adapter(monkeypatch, "preflight", _boom)
    with pytest.raises(GhTransportError):
        _call(milknado_github_roadmap_import, owner="acme", number=7, project_root=str(tmp_path))


def test_bind_tool_projects_roadmap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from milknado.domains.wiki.importer import import_roadmap
    from milknado.mcp._core import open_graph, resolve_project_root

    _seed_wiki(tmp_path)
    root = resolve_project_root(str(tmp_path))
    graph, _cfg = open_graph(root)
    import_roadmap(root / ".hallouminate" / "wiki", "demo", graph)
    graph.close()

    _stub_adapter(monkeypatch, "preflight", lambda: None)
    _stub_adapter(monkeypatch, "project_view", lambda _o, _n: GithubProject(id="PVT_1", title=""))
    _stub_adapter(monkeypatch, "item_list", lambda _o, _n: [])
    _stub_adapter(monkeypatch, "issue_create", lambda *_a: "https://x/1")
    _stub_adapter(monkeypatch, "item_add", lambda _o, _n, _u: "PVTI_1")
    _stub_adapter(monkeypatch, "field_list", lambda _o, _n: [])
    _stub_adapter(monkeypatch, "field_create", lambda *_a: "F")
    _stub_adapter(monkeypatch, "field_create_text", lambda *_a: "F2")

    result = _call(milknado_github_roadmap_bind, roadmap_slug="demo", project_root=str(tmp_path))
    assert result["issues_created"] == 1
    assert result["field_created"] is True


def test_export_tool_after_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_import(monkeypatch)
    _call(milknado_github_roadmap_import, owner="acme", number=7, project_root=str(tmp_path))

    _stub_adapter(monkeypatch, "preflight", lambda: None)
    _stub_adapter(monkeypatch, "project_view", lambda _o, _n: GithubProject(id="PVT_1", title=""))
    _stub_adapter(
        monkeypatch,
        "field_list",
        lambda _o, _n: [
            GithubField(
                id="F_s",
                name="Milknado Status",
                options=({"id": "o", "name": "Pending"},),
            ),
            GithubField(id="F_h", name="Milknado Harvest"),
        ],
    )
    _stub_adapter(
        monkeypatch,
        "item_list",
        lambda _o, _n: [GithubItem(id="PVTI_1", url="https://x/1")],
    )
    _stub_adapter(monkeypatch, "item_edit", lambda *_a, **_k: None)
    _stub_adapter(monkeypatch, "issue_edit_body", lambda *_a: None)

    result = _call(
        milknado_github_roadmap_export, owner="acme", number=7, project_root=str(tmp_path)
    )
    assert result["goals_exported"] == 1
    assert result["bodies_overwritten"] == 0  # github-origin: body never touched


def test_export_tool_before_import_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_adapter(monkeypatch, "preflight", lambda: None)
    _stub_adapter(
        monkeypatch, "project_view", lambda _o, _n: GithubProject(id="PVT_unbound", title="")
    )
    with pytest.raises(LookupError):
        _call(milknado_github_roadmap_export, owner="acme", number=7, project_root=str(tmp_path))
