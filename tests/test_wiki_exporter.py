"""export_roadmap: harvest milknado execution state back into wiki goal files.

Pins the membrane guarantee (Intent/Acceptance byte-identical, only the harvest
block + status/last_synced change), the orphan-goal file-creation path, the
task-status rollup, and that a missing `hallouminate` CLI is non-fatal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeKind, NodeSpec, RunResult
from milknado.domains.graph import MikadoGraph
from milknado.domains.wiki._locate import RoadmapPathError
from milknado.domains.wiki._serialize import (
    HARVEST_END,
    HARVEST_START,
    extract_section,
    load_frontmatter,
)
from milknado.domains.wiki.exporter import export_roadmap, resolve_roadmap_node
from milknado.domains.wiki.importer import import_roadmap
from milknado.domains.wiki.ports import WikiIndexResult, WikiIndexStatus

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


class StubIndexer:
    def __init__(self, result: WikiIndexResult | None = None) -> None:
        self.result = result or WikiIndexResult(WikiIndexStatus.REFRESHED)
        self.roots: list[Path] = []

    def refresh(self, root: Path) -> WikiIndexResult:
        self.roots.append(root)
        return self.result


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
        export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
        after = _goal_path(wiki_root).read_text()
        assert extract_section(before, "Intent") == extract_section(after, "Intent")
        assert extract_section(before, "Acceptance") == extract_section(after, "Acceptance")

    def test_status_and_harvest_updated(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        graph.mark_running(goal_id)
        graph.mark_done(goal_id)
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
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
        export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
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

    def test_symlinked_goal_cannot_overwrite_outside_wiki(
        self, wiki_root: Path, graph: MikadoGraph, tmp_path: Path
    ) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_path = _goal_path(wiki_root)
        outside = tmp_path / "outside.md"
        outside.write_text(GOAL_MD)
        goal_path.unlink()
        goal_path.symlink_to(outside)

        with pytest.raises(RoadmapPathError, match="symlinked roadmap path"):
            export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)

        assert outside.read_text() == GOAL_MD


class TestTaskRollup:
    def test_counts_done_and_failed_tasks(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        t1 = graph.add_node("t1", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
        t2 = graph.add_node("t2", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
        graph.mark_running(t1.id)
        graph.mark_done(t1.id)
        graph.mark_failed(t2.id)
        export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
        text = _goal_path(wiki_root).read_text()
        assert "tasks: 1 done / 1 failed" in text

    def test_summary_includes_deposit_result(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        task = graph.add_node("t1", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
        graph.start_run("run-1", task.id, "/tmp/log", NOW, None)
        graph.finish_run(
            "run-1", RunResult(status="done", exit_code=0, timed_out=False, ended_at=NOW)
        )
        graph.deposit_run_message("run-1", "result", "shipped the exporter", NOW)
        graph.mark_running(task.id)
        graph.mark_done(task.id)
        export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
        text = _goal_path(wiki_root).read_text()
        assert "shipped the exporter" in text

    def test_branch_links_rendered(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        task = graph.add_node("t1", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
        graph.mark_running(task.id, worktree_path="/wt", branch_name="claude/t1", run_id="r1")
        export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
        text = _goal_path(wiki_root).read_text()
        assert "claude/t1" in text


class TestOrphanGoal:
    def test_creates_file_for_goal_node_without_file(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        orphan = graph.add_node(
            "Discovered goal", parent_id=result.roadmap_node_id, spec=NodeSpec(kind=NodeKind.GOAL)
        )
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
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
        graph.add_node(
            "Discovered goal", parent_id=result.roadmap_node_id, spec=NodeSpec(kind=NodeKind.GOAL)
        )
        export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
        second = export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
        # second export rewrites the now-existing file, does not create again.
        assert second.files_created == 0

    def test_duplicate_description_orphans_get_distinct_files(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        # L1: two discovered goals with the SAME description must not overwrite
        # each other's file or collide on the UNIQUE(wiki_ref) index.
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        graph.add_node(
            "Same name", parent_id=result.roadmap_node_id, spec=NodeSpec(kind=NodeKind.GOAL)
        )
        graph.add_node(
            "Same name", parent_id=result.roadmap_node_id, spec=NodeSpec(kind=NodeKind.GOAL)
        )
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
        assert r.files_created == 2
        roadmap_dir = wiki_root / "roadmaps" / ROADMAP_SLUG
        same_name_files = sorted(p.name for p in roadmap_dir.glob("same-name*.md"))
        assert len(same_name_files) == 2  # base + numeric-suffix-disambiguated

    def test_orphan_file_has_membrane_structure(self, wiki_root: Path, graph: MikadoGraph) -> None:
        # crit 6: a created orphan file is a valid round-trip unit — goal
        # frontmatter, the human-owned Intent/Acceptance scaffold, and the
        # harvest markers must all be present or the next import/export breaks.
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        graph.add_node(
            "Discovered goal", parent_id=result.roadmap_node_id, spec=NodeSpec(kind=NodeKind.GOAL)
        )
        export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
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


def test_export_returns_injected_index_failure_without_losing_file_write(
    wiki_root: Path, graph: MikadoGraph
) -> None:
    imported = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
    graph.mark_running(imported.goal_node_ids["wire-export"])
    graph.mark_done(imported.goal_node_ids["wire-export"])
    indexer = StubIndexer(WikiIndexResult(WikiIndexStatus.FAILED, "database locked at /tmp/index"))

    result = export_roadmap(graph, imported.roadmap_node_id, wiki_root, indexer, now=NOW)

    assert result.files_written == 1
    assert result.index == WikiIndexResult(WikiIndexStatus.FAILED, "database locked at /tmp/index")
    assert indexer.roots == [wiki_root]
    assert "status: done" in _goal_path(wiki_root).read_text()


class TestExportGuards:
    def test_non_roadmap_node_raises(self, wiki_root: Path, graph: MikadoGraph) -> None:
        node = graph.add_node("not a roadmap")
        with pytest.raises(ValueError, match="roadmap"):
            export_roadmap(graph, node.id, wiki_root, StubIndexer(), now=NOW)

    def test_roadmap_without_wiki_ref_raises(self, wiki_root: Path, graph: MikadoGraph) -> None:
        node = graph.add_node("rm", spec=NodeSpec(kind=NodeKind.ROADMAP))
        with pytest.raises(ValueError, match="wiki_ref"):
            export_roadmap(graph, node.id, wiki_root, StubIndexer(), now=NOW)

    def test_export_raises_when_wiki_files_deleted_after_import(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        # remove the entire roadmap dir to simulate files deleted post-import
        import shutil as _shutil

        _shutil.rmtree(wiki_root / "roadmaps" / ROADMAP_SLUG)
        with pytest.raises(FileNotFoundError):
            export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)

    def test_resolve_rejects_unsafe_slug(self, wiki_root: Path, graph: MikadoGraph) -> None:
        # L2: the export resolve path validates the slug before joining it too.
        with pytest.raises(ValueError, match="unsafe roadmap slug"):
            resolve_roadmap_node(graph, wiki_root, "../escape")


class TestFlatLayoutExport:
    """Export round-trips the flat `wiki_root/<slug>/` layout."""

    @pytest.fixture()
    def flat_wiki_root(self, tmp_path: Path) -> Path:
        root = tmp_path / ".hallouminate" / "wiki"
        d = root / ROADMAP_SLUG
        d.mkdir(parents=True)
        (d / "index.md").write_text(
            "---\nkind: roadmap\nslug: demo-roadmap\ncreated: 2026-06-01\nstatus: pending\n"
            "---\n# Demo Roadmap\n\nVision prose.\n"
        )
        (d / "wire-export.md").write_text(GOAL_MD)
        return root

    def test_flat_layout_import_then_export(
        self, flat_wiki_root: Path, graph: MikadoGraph
    ) -> None:
        result = import_roadmap(flat_wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        graph.mark_running(goal_id)
        graph.mark_done(goal_id)
        r = export_roadmap(graph, result.roadmap_node_id, flat_wiki_root, StubIndexer(), now=NOW)
        assert r.files_written == 1
        goal_path = flat_wiki_root / ROADMAP_SLUG / "wire-export.md"
        text = goal_path.read_text()
        assert "status: done" in text

    def test_flat_layout_export_prose_byte_identical(
        self, flat_wiki_root: Path, graph: MikadoGraph
    ) -> None:
        goal_path = flat_wiki_root / ROADMAP_SLUG / "wire-export.md"
        before = goal_path.read_text()
        result = import_roadmap(flat_wiki_root, ROADMAP_SLUG, graph)
        graph.mark_running(result.goal_node_ids["wire-export"])
        graph.mark_done(result.goal_node_ids["wire-export"])
        export_roadmap(graph, result.roadmap_node_id, flat_wiki_root, StubIndexer(), now=NOW)
        after = goal_path.read_text()
        assert _human_region(after) == _human_region(before)
        assert after.split(HARVEST_END, 1)[1] == before.split(HARVEST_END, 1)[1]

    def test_resolve_roadmap_node_flat_layout(
        self, flat_wiki_root: Path, graph: MikadoGraph
    ) -> None:
        import_roadmap(flat_wiki_root, ROADMAP_SLUG, graph)
        node = resolve_roadmap_node(graph, flat_wiki_root, ROADMAP_SLUG)
        from milknado.domains.common import NodeKind

        assert node.kind == NodeKind.ROADMAP

    def test_nested_layout_export_regression(self, wiki_root: Path, graph: MikadoGraph) -> None:
        # regression: nested layout export must be unaffected
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        graph.mark_running(goal_id)
        graph.mark_done(goal_id)
        r = export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
        assert r.files_written == 1
        text = _goal_path(wiki_root).read_text()
        assert "status: done" in text

    def test_export_uses_nested_when_both_layouts_exist(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        # When both roadmaps/<slug>/ and <slug>/ have an index.md, import picks
        # the nested layout.  Export's _locate_roadmap_dir must resolve the same
        # nested dir so the harvest block lands in the right file.
        # Create a flat dir with a DIFFERENT goal file to prove export wrote
        # to the nested dir, not the flat one.
        flat_dir = wiki_root / ROADMAP_SLUG
        flat_dir.mkdir(parents=True)
        (flat_dir / "index.md").write_text(
            "---\nkind: roadmap\nslug: demo-roadmap\ncreated: 2026-06-01\nstatus: pending\n"
            "---\n# Demo Roadmap (flat copy)\n\nShould not be touched.\n"
        )
        (flat_dir / "flat-only-goal.md").write_text(
            "---\nkind: goal\ncreated: 2026-06-04\nstatus: pending\n---\n# Flat Goal\n"
        )
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        goal_id = result.goal_node_ids["wire-export"]
        graph.mark_running(goal_id)
        graph.mark_done(goal_id)
        export_roadmap(graph, result.roadmap_node_id, wiki_root, StubIndexer(), now=NOW)
        # The nested goal file is updated
        nested_text = _goal_path(wiki_root).read_text()
        assert "status: done" in nested_text
        # The flat dir's index is byte-identical — export never touched it
        assert (
            (flat_dir / "index.md")
            .read_text()
            .startswith(
                "---\nkind: roadmap\nslug: demo-roadmap\ncreated: 2026-06-01\nstatus: pending\n"
            )
        )
