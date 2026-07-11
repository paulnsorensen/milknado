"""`milknado github-roadmap import|bind|export` CLI veneer.

Exercises the veneer roundtrip and its fail-fast catch shape; the slice gh_*
functions are monkeypatched so nothing shells out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from milknado.adapters.gh import GhTransportError
from milknado.cli import app
from milknado.domains.github import bind as bind_mod
from milknado.domains.github import exporter as exp_mod
from milknado.domains.github import importer as imp_mod

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
"""


def _seed_wiki(project_root: Path) -> None:
    d = project_root / ".hallouminate" / "wiki" / "roadmaps" / "demo"
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "g1.md").write_text(GOAL_MD)


def test_import_command_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(imp_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(imp_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_1", "title": "RM"})
    monkeypatch.setattr(imp_mod, "gh_item_list", lambda _o, _n: [{"id": "PVTI_1", "title": "g"}])
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

    monkeypatch.setattr(imp_mod, "gh_preflight", _boom)
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

    monkeypatch.setattr(bind_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(bind_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_1"})
    monkeypatch.setattr(bind_mod, "gh_issue_create", lambda o, r, _t, _b: f"https://x/{o}/{r}/1")
    monkeypatch.setattr(bind_mod, "gh_item_add", lambda _o, _n, _u: "PVTI_1")
    monkeypatch.setattr(bind_mod, "gh_field_list", lambda _o, _n: [])
    monkeypatch.setattr(bind_mod, "gh_field_create", lambda *_a: {"id": "F"})
    monkeypatch.setattr(bind_mod, "gh_field_create_text", lambda *_a: {"id": "F2"})

    argv = ["github-roadmap", "bind", "demo", "--project-root", str(tmp_path)]
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert "Bound roadmap demo" in result.output
    assert "field_created=True" in result.output


def test_export_command_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # github-origin: import creates a bound roadmap, export harvests it.
    monkeypatch.setattr(imp_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(imp_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_1", "title": "RM"})
    monkeypatch.setattr(imp_mod, "gh_item_list", lambda _o, _n: [{"id": "PVTI_1", "title": "g"}])
    imp = runner.invoke(
        app, ["github-roadmap", "import", "acme", "7", "--project-root", str(tmp_path)]
    )
    assert imp.exit_code == 0, imp.output

    monkeypatch.setattr(exp_mod, "gh_preflight", lambda: None)
    monkeypatch.setattr(exp_mod, "gh_project_view", lambda _o, _n: {"id": "PVT_1"})
    monkeypatch.setattr(
        exp_mod,
        "gh_field_list",
        lambda _o, _n: [
            {
                "id": "F_status",
                "name": "Milknado Status",
                "options": [{"id": "o", "name": "Pending"}],
            },
            {"id": "F_harvest", "name": "Milknado Harvest"},
        ],
    )
    monkeypatch.setattr(
        exp_mod, "gh_item_list", lambda _o, _n: [{"id": "PVTI_1", "url": "https://x/1"}]
    )
    monkeypatch.setattr(exp_mod, "gh_item_edit", lambda *_a, **_k: None)
    monkeypatch.setattr(exp_mod, "gh_issue_edit_body", lambda *_a: None)

    result = runner.invoke(
        app, ["github-roadmap", "export", "acme", "7", "--project-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "Exported project acme/7" in result.output
    assert "1 goals" in result.output
