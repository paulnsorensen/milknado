"""MCP veneer for the GitHub Projects crossover: import + bind + export tools."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from milknado.adapters.gh import GhProjectAdapter, GhTransportError
from milknado.domains.github import GithubField, GithubFieldOption, GithubItem, GithubProject
from milknado.mcp._core import Response
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

## Acceptance
- done

"""


def _preflight_ok() -> None:
    return None


def _project_bound(_owner: str, _number: int) -> GithubProject:
    return GithubProject(id="PVT_1", title="RM")


def _project_empty(_owner: str, _number: int) -> GithubProject:
    return GithubProject(id="PVT_1", title="")


def _project_unbound(_owner: str, _number: int) -> GithubProject:
    return GithubProject(id="PVT_unbound", title="")


def _items_for_import(_owner: str, _number: int) -> list[GithubItem]:
    return [GithubItem(id="PVTI_1", title="g")]


def _items_for_export(_owner: str, _number: int) -> list[GithubItem]:
    return [GithubItem(id="PVTI_1", url="https://x/1")]


def _no_items(_owner: str, _number: int) -> list[GithubItem]:
    return []


def _no_fields(_owner: str, _number: int) -> list[GithubField]:
    return []


def _fields_for_export(_owner: str, _number: int) -> list[GithubField]:
    return [
        GithubField(
            id="F_s",
            name="Milknado Status",
            options=(GithubFieldOption(id="o", name="Pending"),),
        ),
        GithubField(id="F_h", name="Milknado Harvest"),
    ]


def _issue_url(*_args: object) -> str:
    return "https://x/1"


def _item_id(_owner: str, _number: int, _url: str) -> str:
    return "PVTI_1"


def _field_id(*_args: object) -> str:
    return "F"


def _field_text_id(*_args: object) -> str:
    return "F2"


def _void(*_args: object, **_kwargs: object) -> None:
    return None


def _call(tool: object, **kwargs: object) -> Response:
    fn = cast(Callable[..., Response], getattr(tool, "fn", tool))
    return fn(**kwargs)


def _stub_adapter(
    monkeypatch: pytest.MonkeyPatch, name: str, value: Callable[..., object]
) -> None:
    monkeypatch.setattr(GhProjectAdapter, name, staticmethod(value))


def _seed_wiki(project_root: Path) -> None:
    d = project_root / ".hallouminate" / "wiki" / "roadmaps" / "demo"
    d.mkdir(parents=True)
    _ = (d / "index.md").write_text(INDEX_MD)
    _ = (d / "g1.md").write_text(GOAL_MD)


def _stub_import(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_adapter(monkeypatch, "preflight", _preflight_ok)
    _stub_adapter(monkeypatch, "project_view", _project_bound)
    _stub_adapter(monkeypatch, "item_list", _items_for_import)


def test_import_tool_seeds_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_import(monkeypatch)
    result = _call(
        milknado_github_roadmap_import, owner="acme", number=7, project_root=str(tmp_path)
    )
    assert result["created"] == 2
    goal_node_ids = cast(dict[str, int], result["goal_node_ids"])
    assert set(goal_node_ids) == {"PVTI_1"}
    assert isinstance(goal_node_ids["PVTI_1"], int)


def test_import_tool_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> None:
        raise GhTransportError("gh auth refresh -s project")

    _stub_adapter(monkeypatch, "preflight", _boom)
    with pytest.raises(GhTransportError):
        _ = _call(
            milknado_github_roadmap_import, owner="acme", number=7, project_root=str(tmp_path)
        )


def test_bind_tool_projects_roadmap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from milknado.domains.wiki.importer import import_roadmap
    from milknado.mcp._core import open_graph, resolve_project_root

    _seed_wiki(tmp_path)
    root = resolve_project_root(str(tmp_path))
    graph, _cfg = open_graph(root)
    _ = import_roadmap(root / ".hallouminate" / "wiki", "demo", graph)
    graph.close()

    _stub_adapter(monkeypatch, "preflight", _preflight_ok)
    _stub_adapter(monkeypatch, "project_view", _project_empty)
    _stub_adapter(monkeypatch, "item_list", _no_items)
    _stub_adapter(monkeypatch, "issue_create", _issue_url)
    _stub_adapter(monkeypatch, "item_add", _item_id)
    _stub_adapter(monkeypatch, "field_list", _no_fields)
    _stub_adapter(monkeypatch, "field_create", _field_id)
    _stub_adapter(monkeypatch, "field_create_text", _field_text_id)

    result = _call(milknado_github_roadmap_bind, roadmap_slug="demo", project_root=str(tmp_path))
    assert result["issues_created"] == 1
    assert result["field_created"] is True


def test_export_tool_after_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_import(monkeypatch)
    _ = _call(milknado_github_roadmap_import, owner="acme", number=7, project_root=str(tmp_path))

    _stub_adapter(monkeypatch, "preflight", _preflight_ok)
    _stub_adapter(monkeypatch, "project_view", _project_empty)
    _stub_adapter(monkeypatch, "field_list", _fields_for_export)
    _stub_adapter(monkeypatch, "item_list", _items_for_export)
    _stub_adapter(monkeypatch, "item_edit", _void)
    _stub_adapter(monkeypatch, "issue_edit_body", _void)

    result = _call(
        milknado_github_roadmap_export, owner="acme", number=7, project_root=str(tmp_path)
    )
    assert result["goals_exported"] == 1
    assert result["bodies_overwritten"] == 0  # github-origin: body never touched


def test_export_tool_before_import_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_adapter(monkeypatch, "preflight", _preflight_ok)
    _stub_adapter(monkeypatch, "project_view", _project_unbound)
    with pytest.raises(LookupError):
        _ = _call(
            milknado_github_roadmap_export, owner="acme", number=7, project_root=str(tmp_path)
        )
