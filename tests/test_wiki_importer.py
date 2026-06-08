"""import_roadmap: parse a wiki roadmap dir into milknado roadmap + goal nodes.

Pins the acceptance criteria: one roadmap node + one goal node per file (no task
nodes), prereqs wired, deterministic wiki_ref across separate dbs, idempotent
re-import, the autoincrement id never leaking into a wiki file, and fail-fast on
a missing dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeKind
from milknado.domains.graph import MikadoGraph
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

GOAL_A = """---
kind: goal
slug: define-schema
roadmap: demo-roadmap
created: 2026-06-02
status: pending
flavor: implement
prereqs: []
---
# Define the schema

## Intent
Lay the schema groundwork.

## Acceptance
- columns exist
"""

GOAL_B = """---
kind: goal
slug: wire-export
roadmap: demo-roadmap
created: 2026-06-03
status: pending
flavor: implement
prereqs: [define-schema]
---
# Wire the export

## Intent
Export to the wiki.

## Acceptance
- harvest written
"""


def _make_roadmap(wiki_root: Path) -> Path:
    d = wiki_root / "roadmaps" / ROADMAP_SLUG
    d.mkdir(parents=True)
    (d / "index.md").write_text(INDEX_MD)
    (d / "define-schema.md").write_text(GOAL_A)
    (d / "wire-export.md").write_text(GOAL_B)
    return d


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    root = tmp_path / ".hallouminate" / "wiki"
    root.mkdir(parents=True)
    _make_roadmap(root)
    return root


class TestImportStructure:
    def test_creates_roadmap_and_goal_nodes_only(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        nodes = graph.get_all_nodes()
        kinds = [n.kind for n in nodes]
        assert kinds.count(NodeKind.ROADMAP) == 1
        assert kinds.count(NodeKind.GOAL) == 2
        assert kinds.count(NodeKind.TASK) == 0
        assert result.created_count == 3
        assert set(result.goal_node_ids) == {"define-schema", "wire-export"}

    def test_goal_nodes_parented_under_roadmap(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        children = graph.get_children(result.roadmap_node_id)
        assert {c.kind for c in children} == {NodeKind.GOAL}
        assert len(children) == 2

    def test_prereqs_wired_as_edges(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        wire_id = result.goal_node_ids["wire-export"]
        schema_id = result.goal_node_ids["define-schema"]
        prereq_ids = {c.id for c in graph.get_children(wire_id)}
        assert schema_id in prereq_ids

    def test_description_from_h1_title(self, wiki_root: Path, graph: MikadoGraph) -> None:
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        schema = graph.get_node(result.goal_node_ids["define-schema"])
        assert schema is not None
        assert schema.description == "Define the schema"

    def test_empty_roadmap_creates_only_roadmap_node(
        self, tmp_path: Path, graph: MikadoGraph
    ) -> None:
        # crit 1 boundary: a roadmap dir with index.md and zero goal files
        # seeds exactly one roadmap node and no goals — not an error.
        root = tmp_path / ".hallouminate" / "wiki"
        d = root / "roadmaps" / ROADMAP_SLUG
        d.mkdir(parents=True)
        (d / "index.md").write_text(INDEX_MD)
        result = import_roadmap(root, ROADMAP_SLUG, graph)
        assert result.created_count == 1
        assert result.goal_node_ids == {}
        kinds = [n.kind for n in graph.get_all_nodes()]
        assert kinds.count(NodeKind.ROADMAP) == 1
        assert kinds.count(NodeKind.GOAL) == 0


class TestDeterminism:
    def test_same_ref_across_separate_dbs(self, wiki_root: Path, tmp_path: Path) -> None:
        g1 = MikadoGraph(tmp_path / "db1.db")
        g2 = MikadoGraph(tmp_path / "db2.db")
        try:
            r1 = import_roadmap(wiki_root, ROADMAP_SLUG, g1)
            r2 = import_roadmap(wiki_root, ROADMAP_SLUG, g2)
            n1 = g1.get_node(r1.goal_node_ids["wire-export"])
            n2 = g2.get_node(r2.goal_node_ids["wire-export"])
            assert n1 is not None and n2 is not None
            assert n1.wiki_ref is not None
            assert n1.wiki_ref == n2.wiki_ref
        finally:
            g1.close()
            g2.close()

    def test_roadmap_ref_deterministic_across_dbs(self, wiki_root: Path, tmp_path: Path) -> None:
        # crit 2: the roadmap node's key is content-addressed too, so the same
        # roadmap resolves to the identical wiki_ref in independent dbs — this is
        # what lets export find the milknado roadmap from the wiki slug alone.
        g1 = MikadoGraph(tmp_path / "rdb1.db")
        g2 = MikadoGraph(tmp_path / "rdb2.db")
        try:
            r1 = import_roadmap(wiki_root, ROADMAP_SLUG, g1)
            r2 = import_roadmap(wiki_root, ROADMAP_SLUG, g2)
            rm1 = g1.get_node(r1.roadmap_node_id)
            rm2 = g2.get_node(r2.roadmap_node_id)
            assert rm1 is not None and rm2 is not None
            assert rm1.wiki_ref is not None
            assert rm1.wiki_ref == rm2.wiki_ref
        finally:
            g1.close()
            g2.close()

    def test_autoincrement_id_never_written_to_wiki(
        self, wiki_root: Path, graph: MikadoGraph
    ) -> None:
        # With `created` already present, import writes nothing back, so every
        # file is byte-identical — the autoincrement id cannot have leaked.
        roadmap_dir = wiki_root / "roadmaps" / ROADMAP_SLUG
        before = {p.name: p.read_text() for p in roadmap_dir.glob("*.md")}
        import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        after = {p.name: p.read_text() for p in roadmap_dir.glob("*.md")}
        assert before == after


class TestIdempotency:
    def test_reimport_creates_no_duplicates(self, wiki_root: Path, graph: MikadoGraph) -> None:
        first = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        before = len(graph.get_all_nodes())
        second = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        after = len(graph.get_all_nodes())
        assert before == after == 3
        assert second.created_count == 0
        assert second.reused_count == 3
        assert first.roadmap_node_id == second.roadmap_node_id

    def test_reimport_does_not_duplicate_edges(self, wiki_root: Path, graph: MikadoGraph) -> None:
        import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        result = import_roadmap(wiki_root, ROADMAP_SLUG, graph)
        wire_id = result.goal_node_ids["wire-export"]
        # One prereq edge only, not two.
        assert len(graph.get_children(wire_id)) == 1


class TestCreatedStamping:
    def test_stamps_created_when_absent(self, tmp_path: Path, graph: MikadoGraph) -> None:
        root = tmp_path / ".hallouminate" / "wiki"
        gdir = root / "roadmaps" / ROADMAP_SLUG
        gdir.mkdir(parents=True)
        (gdir / "index.md").write_text(INDEX_MD)
        (gdir / "no-date.md").write_text(
            "---\nkind: goal\nslug: no-date\nroadmap: demo-roadmap\nstatus: pending\n"
            "prereqs: []\n---\n# No date\n\n## Intent\nx\n"
        )
        before = (gdir / "no-date.md").read_text()
        result = import_roadmap(root, ROADMAP_SLUG, graph, today="2026-06-09")
        assert "no-date" in result.stamped
        after = (gdir / "no-date.md").read_text()
        assert "created: 2026-06-09" in after
        # crit 4: stamping is the only write-back path, and the autoincrement
        # node id must not leak through it. The ONLY line added is the `created`
        # stamp — nothing else (least of all a node id) is written to the file.
        added = set(after.splitlines()) - set(before.splitlines())
        assert added == {"created: 2026-06-09"}


class TestFailFast:
    def test_missing_roadmap_dir_raises(self, tmp_path: Path, graph: MikadoGraph) -> None:
        root = tmp_path / ".hallouminate" / "wiki"
        root.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="roadmap"):
            import_roadmap(root, "ghost-roadmap", graph)

    def test_missing_wiki_root_raises(self, tmp_path: Path, graph: MikadoGraph) -> None:
        with pytest.raises(FileNotFoundError):
            import_roadmap(tmp_path / "nope", ROADMAP_SLUG, graph)
