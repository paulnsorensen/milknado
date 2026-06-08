"""export_roadmap: harvest milknado execution state back into wiki goal files.

Pins the membrane guarantee (Intent/Acceptance byte-identical, only the harvest
block + status/last_synced change), the orphan-goal file-creation path, the
task-status rollup, and that a missing `hallouminate` CLI is non-fatal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeKind
from milknado.domains.graph import MikadoGraph
from milknado.domains.wiki._serialize import (
    HARVEST_END,
    HARVEST_START,
    extract_section,
    load_frontmatter,
)
from milknado.domains.wiki.exporter import export_roadmap, resolve_roadmap_node
from milknado.domains.wiki.importer import import_roadmap

ROADMAP_SLUG = "demo-roadmap"

INDEX_MD = """---
kind: roadmap
slug: demo-roadmap
created: 2026-06-01
status: pending
---
# Demo Roadmap

Vision prose.
"""

GOAL_MD = """---
kind: goal
slug: wire-export
roadmap: demo-roadmap
created: 2026-06-03
status: pending
flavor: implement
prereqs: []
---
# Wire the export

## Intent
Export to the wiki without clobbering human text.

## Acceptance
- harvest block written
- intent untouched

## Outcome
<!-- milknado:harvest:start -->
result: pending
<!-- milknado:harvest:end -->
"""

NOW = "2026-06-08T12:00:00Z"


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    d = tmp_path / ".hallouminate" / "wiki" / "roadmaps" / ROADMAP_SLUG
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "wire-export.md").write_text(GOAL_MD)
    return tmp_path / ".hallouminate" / "wiki"


def _goal_path(wiki_root: Path) -> Path:
    return wiki_root / "roadmaps" / ROADMAP_SLUG / "wire-export.md"


def _human_region(text: str) -> str:
    """The raw bytes a human owns: everything after the frontmatter fence up to
    the harvest-start marker (title + Intent + Acceptance + the `## Outcome`
    heading). Unlike extract_section, this preserves whitespace and blank lines,
    so it catches any byte the exporter reflows."""
    body = text.split("\n---\n", 1)[1]
    return body.split(HARVEST_START, 1)[0]


class TestExportMembrane:
    def test_intent_and_acceptance_byte_identical(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        before = _goal_path(wiki_root).read_text()
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        graph.mark_running(result.goal_node_ids["wire-export"])
        graph.mark_done(result.goal_node_ids["wire-export"])
        export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        after = _goal_path(wiki_root).read_text()
        assert extract_section(before, "Intent") == extract_section(after, "Intent")
        assert extract_section(before, "Acceptance") == extract_section(after, "Acceptance")

    def test_status_and_harvest_updated(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        graph.mark_running(goal_id)
        graph.mark_done(goal_id)
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        text = _goal_path(wiki_root).read_text()
        assert "status: done" in text
        assert "result: done" in text
        assert "result: pending" not in text
        assert r.files_written == 1
        assert r.files_created == 0

    def test_human_region_and_tail_byte_identical(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        # crit 5, hardened: a raw-byte check (not whitespace-stripped sections).
        # Everything outside the harvest block — the human-owned body and the
        # post-marker tail — must be verbatim-identical across an export cycle.
        before = _goal_path(wiki_root).read_text()
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        graph.mark_running(result.goal_node_ids["wire-export"])
        graph.mark_done(result.goal_node_ids["wire-export"])
        export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        after = _goal_path(wiki_root).read_text()
        assert _human_region(after) == _human_region(before)
        assert after.split(HARVEST_END, 1)[1] == before.split(HARVEST_END, 1)[1]
        # Only status/last_synced frontmatter may change; the rest is untouched.
        for line in (
            "kind: goal",
            "slug: wire-export",
            "roadmap: demo-roadmap",
            "created: 2026-06-03",
            "flavor: implement",
            "prereqs: []",
        ):
            assert line in after


class TestTaskRollup:
    def test_counts_done_and_failed_tasks(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        t1 = graph.add_node("t1", parent_id=goal_id, kind=NodeKind.TASK)
        t2 = graph.add_node("t2", parent_id=goal_id, kind=NodeKind.TASK)
        graph.mark_running(t1.id)
        graph.mark_done(t1.id)
        graph.mark_failed(t2.id)
        export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        text = _goal_path(wiki_root).read_text()
        assert "tasks: 1 done / 1 failed" in text

    def test_summary_includes_deposit_result(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        task = graph.add_node("t1", parent_id=goal_id, kind=NodeKind.TASK)
        graph.start_run("run-1", task.id, "/tmp/log", NOW, None)
        graph.finish_run("run-1", status="done", exit_code=0, timed_out=False, ended_at=NOW)
        graph.deposit_run_message("run-1", "result", "shipped the exporter", NOW)
        graph.mark_running(task.id)
        graph.mark_done(task.id)
        export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        text = _goal_path(wiki_root).read_text()
        assert "shipped the exporter" in text

    def test_branch_links_rendered(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        task = graph.add_node("t1", parent_id=goal_id, kind=NodeKind.TASK)
        graph.mark_running(task.id, worktree_path="/wt", branch_name="claude/t1", run_id="r1")
        export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        text = _goal_path(wiki_root).read_text()
        assert "claude/t1" in text


class TestOrphanGoal:
    def test_creates_file_for_goal_node_without_file(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        orphan = graph.add_node(
            "Discovered goal", parent_id=result.roadmap_node_id, kind=NodeKind.GOAL
        )
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        new_path = wiki_root / "roadmaps" / ROADMAP_SLUG / "discovered-goal.md"
        assert new_path.exists()
        assert r.files_created == 1
        # node gains a wiki_ref so re-export locates the file (round-trips).
        reloaded = graph.get_node(orphan.id)
        assert reloaded is not None
        assert reloaded.wiki_ref is not None

    def test_orphan_file_round_trips_on_reexport(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        graph.add_node("Discovered goal", parent_id=result.roadmap_node_id, kind=NodeKind.GOAL)
        export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        second = export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        # second export rewrites the now-existing file, does not create again.
        assert second.files_created == 0

    def test_duplicate_description_orphans_get_distinct_files(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        # L1: two discovered goals with the SAME description must not overwrite
        # each other's file or collide on the UNIQUE(wiki_ref) index.
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        graph.add_node("Same name", parent_id=result.roadmap_node_id, kind=NodeKind.GOAL)
        graph.add_node("Same name", parent_id=result.roadmap_node_id, kind=NodeKind.GOAL)
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        assert r.files_created == 2
        roadmap_dir = wiki_root / "roadmaps" / ROADMAP_SLUG
        same_name_files = sorted(p.name for p in roadmap_dir.glob("same-name*.md"))
        assert len(same_name_files) == 2  # base + numeric-suffix-disambiguated

    def test_orphan_file_has_membrane_structure(self, wiki_root: Path, graph: MikadoGraph) -> None:
        # crit 6: a created orphan file is a valid round-trip unit — goal
        # frontmatter, the human-owned Intent/Acceptance scaffold, and the
        # harvest markers must all be present or the next import/export breaks.
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        graph.add_node("Discovered goal", parent_id=result.roadmap_node_id, kind=NodeKind.GOAL)
        export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        text = (wiki_root / "roadmaps" / ROADMAP_SLUG / "discovered-goal.md").read_text()
        fm = load_frontmatter(text)
        assert fm["kind"] == "goal"
        assert fm["slug"] == "discovered-goal"
        assert fm["roadmap"] == ROADMAP_SLUG
        assert "created" in fm
        assert "## Intent" in text
        assert "## Acceptance" in text
        assert HARVEST_START in text
        assert HARVEST_END in text


class TestBestEffortIndex:
    def test_export_succeeds_without_hallouminate_cli(
        self, wiki_root: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _name: None)
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        assert r.index_refreshed is False
        assert r.files_written == 1

    def test_export_succeeds_when_index_subprocess_raises(
        self, wiki_root: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # crit 7: the CLI is on PATH but the call blows up — export must still
        # write every file and report index_refreshed=False (non-fatal).
        import shutil
        import subprocess

        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/hallouminate")

        def _boom(*_args: object, **_kwargs: object) -> object:
            raise OSError("hallouminate exploded")

        monkeypatch.setattr(subprocess, "run", _boom)
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        graph.mark_running(result.goal_node_ids["wire-export"])
        graph.mark_done(result.goal_node_ids["wire-export"])
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        assert r.index_refreshed is False
        assert r.files_written == 1
        assert "status: done" in _goal_path(wiki_root).read_text()

    def test_index_refresh_invokes_cli_in_wiki_root(
        self, wiki_root: Path, graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # crit 7, positive path: when the CLI is present and the call succeeds,
        # export reports index_refreshed=True and runs `hallouminate index` with
        # the wiki root as cwd.
        import shutil
        import subprocess

        calls: list[tuple[list[str], str]] = []
        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/hallouminate")

        def _capture(cmd: list[str], **kwargs: object) -> None:
            calls.append((cmd, str(kwargs.get("cwd"))))

        monkeypatch.setattr(subprocess, "run", _capture)
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, now=NOW)
        assert r.index_refreshed is True
        assert calls == [(["hallouminate", "index"], str(wiki_root))]


class TestExportGuards:
    def test_non_roadmap_node_raises(self, wiki_root: Path, graph: MikadoGraph) -> None:
        node = graph.add_node("not a roadmap")
        with pytest.raises(ValueError, match="roadmap"):
            export_roadmap(graph, node.id, wiki_root, now=NOW)

    def test_roadmap_without_wiki_ref_raises(self, wiki_root: Path, graph: MikadoGraph) -> None:
        node = graph.add_node("rm", kind=NodeKind.ROADMAP)
        with pytest.raises(ValueError, match="wiki_ref"):
            export_roadmap(graph, node.id, wiki_root, now=NOW)

    def test_resolve_rejects_unsafe_slug(self, wiki_root: Path, graph: MikadoGraph) -> None:
        # L2: the export resolve path validates the slug before joining it too.
        with pytest.raises(ValueError, match="unsafe roadmap slug"):
            resolve_roadmap_node(graph, wiki_root, "../escape")
