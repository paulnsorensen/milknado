"""`milknado github-roadmap import|bind|export` CLI veneer.

Exercises the veneer roundtrip and its fail-fast catch shape; the slice gh_*
functions are monkeypatched so nothing shells out.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest
from typer.testing import CliRunner

from milknado.adapters.gh import GhProjectAdapter, GhTransportError
from milknado.cli import app
from milknado.domains.github import GithubField, GithubItem, GithubProject

runner = CliRunner()

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


def _seed_wiki(project_root: Path) -> None:
    d = project_root / ".hallouminate" / "wiki" / "roadmaps" / "demo"
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "g1.md").write_text(GOAL_MD)


def test_import_command_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GhProjectAdapter, "preflight", staticmethod(lambda: None))
    monkeypatch.setattr(
        GhProjectAdapter,
        "project_view",
        staticmethod(lambda _o, _n: GithubProject(id="PVT_1", title="RM")),
    )
    monkeypatch.setattr(
        GhProjectAdapter,
        "item_list",
        staticmethod(lambda _o, _n: [GithubItem(id="PVTI_1", title="g")]),
    )
    result = runner.invoke(
        app, ["github-roadmap", "import", "acme", "7", "--project-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "Imported project acme/7" in result.output
    assert "1 goals" in result.output


def test_import_fail_fast_on_transport_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> None:
        raise GhTransportError("gh auth refresh -s project")

    monkeypatch.setattr(GhProjectAdapter, "preflight", staticmethod(_boom))
    result = runner.invoke(
        app, ["github-roadmap", "import", "acme", "7", "--project-root", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "gh auth refresh -s project" in result.output


def test_bind_command_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_wiki(tmp_path)
    # Seed milknado nodes from the wiki so resolve_roadmap_node finds the roadmap.
    imp = runner.invoke(app, ["roadmap", "import", "demo", "--project-root", str(tmp_path)])
    assert imp.exit_code == 0, imp.output

    monkeypatch.setattr(GhProjectAdapter, "preflight", staticmethod(lambda: None))
    monkeypatch.setattr(
        GhProjectAdapter,
        "project_view",
        staticmethod(lambda _o, _n: GithubProject(id="PVT_1", title="RM")),
    )
    monkeypatch.setattr(GhProjectAdapter, "item_list", staticmethod(lambda _o, _n: []))
    monkeypatch.setattr(
        GhProjectAdapter,
        "issue_create",
        staticmethod(lambda o, r, _t, _b: f"https://x/{o}/{r}/1"),
    )
    monkeypatch.setattr(GhProjectAdapter, "item_add", staticmethod(lambda _o, _n, _u: "PVTI_1"))
    monkeypatch.setattr(GhProjectAdapter, "field_list", staticmethod(lambda _o, _n: []))
    monkeypatch.setattr(GhProjectAdapter, "field_create", staticmethod(lambda *_a: "F"))
    monkeypatch.setattr(GhProjectAdapter, "field_create_text", staticmethod(lambda *_a: "F2"))

    argv = ["github-roadmap", "bind", "demo", "--project-root", str(tmp_path)]
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert "Bound roadmap demo" in result.output
    assert "field_created=True" in result.output


def test_export_command_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # github-origin: import creates a bound roadmap, export harvests it.
    monkeypatch.setattr(GhProjectAdapter, "preflight", staticmethod(lambda: None))
    monkeypatch.setattr(
        GhProjectAdapter,
        "project_view",
        staticmethod(lambda _o, _n: GithubProject(id="PVT_1", title="RM")),
    )
    monkeypatch.setattr(
        GhProjectAdapter,
        "item_list",
        staticmethod(lambda _o, _n: [GithubItem(id="PVTI_1", title="g")]),
    )
    imp = runner.invoke(
        app, ["github-roadmap", "import", "acme", "7", "--project-root", str(tmp_path)]
    )
    assert imp.exit_code == 0, imp.output

    monkeypatch.setattr(GhProjectAdapter, "preflight", staticmethod(lambda: None))
    monkeypatch.setattr(
        GhProjectAdapter,
        "project_view",
        staticmethod(lambda _o, _n: GithubProject(id="PVT_1", title="RM")),
    )
    monkeypatch.setattr(
        GhProjectAdapter,
        "field_list",
        staticmethod(
            lambda _o, _n: [
                msgspec.convert(
                    {
                        "id": "F_status",
                        "name": "Milknado Status",
                        "options": [{"id": "o", "name": "Pending"}],
                    },
                    type=GithubField,
                    strict=True,
                ),
                GithubField(id="F_harvest", name="Milknado Harvest"),
            ]
        ),
    )
    monkeypatch.setattr(
        GhProjectAdapter,
        "item_list",
        staticmethod(lambda _o, _n: [GithubItem(id="PVTI_1", url="https://x/1")]),
    )
    monkeypatch.setattr(GhProjectAdapter, "item_edit", staticmethod(lambda *_a, **_k: None))
    monkeypatch.setattr(GhProjectAdapter, "issue_edit_body", staticmethod(lambda *_a: None))

    result = runner.invoke(
        app, ["github-roadmap", "export", "acme", "7", "--project-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "Exported project acme/7" in result.output
    assert "1 goals" in result.output
