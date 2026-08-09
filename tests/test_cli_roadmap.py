"""milknado roadmap CLI veneer."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from milknado.cli import app

runner = CliRunner()


INDEX_MD = """---
kind: roadmap
slug: demo
created: 2026-06-01
status: pending
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


def _seed(project_root: Path) -> None:
    d = project_root / ".hallouminate" / "wiki" / "roadmaps" / "demo"
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "g1.md").write_text(GOAL_MD)


def test_import_then_export_roundtrip(tmp_path: Path) -> None:
    _seed(tmp_path)
    imp = runner.invoke(app, ["roadmap", "import", "demo", "--project-root", str(tmp_path)])
    assert imp.exit_code == 0, imp.output
    assert "Imported roadmap demo" in imp.output
    assert "2 created" in imp.output  # 1 roadmap node + 1 goal node

    exp = runner.invoke(app, ["roadmap", "export", "demo", "--project-root", str(tmp_path)])
    assert exp.exit_code == 0, exp.output
    assert "Exported roadmap demo" in exp.output
    assert "1 updated" in exp.output


def test_import_missing_roadmap_exits_nonzero(tmp_path: Path) -> None:
    (tmp_path / ".hallouminate" / "wiki").mkdir(parents=True)
    result = runner.invoke(app, ["roadmap", "import", "ghost", "--project-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "roadmap not found" in result.output


def test_export_before_import_exits_nonzero(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = runner.invoke(app, ["roadmap", "export", "demo", "--project-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "not imported" in result.output


def test_import_unknown_prereq_exits_nonzero(tmp_path: Path) -> None:
    # An unknown prereq raises ValueError from import_roadmap; the CLI must map it
    # to a clean exit-1 message, not a raw traceback (M1).
    d = tmp_path / ".hallouminate" / "wiki" / "roadmaps" / "demo"
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "g1.md").write_text(
        "---\nkind: goal\nslug: g1\nroadmap: demo\ncreated: 2026-06-02\n"
        "status: pending\nprereqs: [ghost-goal]\n---\n# Goal one\n"
        "\n## Intent\ndo the thing\n\n## Acceptance\n- done\n"
    )
    result = runner.invoke(app, ["roadmap", "import", "demo", "--project-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "ghost-goal" in result.output


def test_schema_json_and_render_commands(tmp_path: Path) -> None:
    _seed(tmp_path)

    schema = runner.invoke(app, ["roadmap", "schema"])
    assert schema.exit_code == 0, schema.output
    assert '"RoadmapModel"' in schema.output

    output = tmp_path / "roadmap.json"
    result = runner.invoke(
        app,
        ["roadmap", "json", "demo", "--project-root", str(tmp_path), "--out", str(output)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["roadmap"]["slug"] == "demo"
    assert payload["goals"]["g1"]["slug"] == "g1"
    assert payload["edges"] == []

    mermaid = runner.invoke(
        app,
        ["roadmap", "render", "demo", "--project-root", str(tmp_path), "--format", "mermaid"],
    )
    assert mermaid.exit_code == 0, mermaid.output
    assert mermaid.output.startswith("graph TD")

    html = runner.invoke(
        app,
        ["roadmap", "render", "demo", "--project-root", str(tmp_path), "--format", "html"],
    )
    assert html.exit_code == 0, html.output
    assert "<!doctype html>" in html.output


def test_schema_out_write_failure_exits_nonzero(tmp_path: Path) -> None:
    output = tmp_path / "missing" / "schema.json"
    result = runner.invoke(app, ["roadmap", "schema", "--out", str(output)])
    assert result.exit_code == 1


def test_json_missing_roadmap_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["roadmap", "json", "ghost", "--project-root", str(tmp_path)])
    assert result.exit_code == 1


def test_render_missing_roadmap_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["roadmap", "render", "ghost", "--project-root", str(tmp_path)])
    assert result.exit_code == 1


def test_json_rejects_symlinked_roadmap(tmp_path: Path) -> None:
    wiki = tmp_path / ".hallouminate" / "wiki"
    wiki.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (wiki / "roadmaps").mkdir()
    (wiki / "roadmaps" / "demo").symlink_to(outside, target_is_directory=True)
    result = runner.invoke(app, ["roadmap", "json", "demo", "--project-root", str(tmp_path)])
    assert result.exit_code == 1
