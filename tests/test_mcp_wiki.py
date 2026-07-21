"""MCP veneer for the wiki roadmap crossover: import + export tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.mcp.wiki import milknado_roadmap_export, milknado_roadmap_import

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


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


def _seed(project_root: Path) -> None:
    d = project_root / ".hallouminate" / "wiki" / "roadmaps" / "demo"
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "g1.md").write_text(GOAL_MD)


def test_import_tool_seeds_nodes(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = _call(milknado_roadmap_import, roadmap_slug="demo", project_root=str(tmp_path))
    assert result["created"] == 2
    assert "g1" in result["goal_node_ids"]


def test_export_tool_after_import(tmp_path: Path) -> None:
    _seed(tmp_path)
    _call(milknado_roadmap_import, roadmap_slug="demo", project_root=str(tmp_path))
    result = _call(milknado_roadmap_export, roadmap_slug="demo", project_root=str(tmp_path))
    assert result["files_written"] == 1
    assert result["files_created"] == 0


def test_import_missing_roadmap_raises(tmp_path: Path) -> None:
    # crit 8: the veneer fails fast — a missing roadmap dir propagates, no
    # partial graph, no swallowed error returned as success.
    _seed(tmp_path)
    with pytest.raises(FileNotFoundError):
        _call(milknado_roadmap_import, roadmap_slug="ghost", project_root=str(tmp_path))


def test_export_before_import_raises(tmp_path: Path) -> None:
    # crit 8: exporting a roadmap that was never imported into milknado fails
    # fast rather than writing empty harvest blocks.
    _seed(tmp_path)
    with pytest.raises(LookupError):
        _call(milknado_roadmap_export, roadmap_slug="demo", project_root=str(tmp_path))
