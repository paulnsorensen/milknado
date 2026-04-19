from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from milknado.domains.batching import (
    Batch,
    BatchPlan,
    FileChange,
    NewRelationship,
    SymbolRef,
    SymbolSpread,
)
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning.batching_bridge import (
    apply_batches_to_graph,
    run_batching,
)
from milknado.domains.planning.manifest import (
    MANIFEST_VERSION,
    PlanChangeManifest,
)


def _change(cid: str, path: str, description: str = "", **kw: object) -> FileChange:
    kwargs: dict[str, object] = {"id": cid, "path": path, "description": description or cid}
    kwargs.update(kw)
    return FileChange(**kwargs)  # type: ignore[arg-type]


def _manifest(
    *changes: FileChange,
    new_rels: tuple[NewRelationship, ...] = (),
    goal: str = "bridge test goal",
    goal_summary: str = "bridge test goal summary",
    spec_path: str | None = None,
) -> PlanChangeManifest:
    return PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        changes=changes,
        new_relationships=new_rels,
        goal=goal,
        goal_summary=goal_summary,
        spec_path=spec_path,
    )


@pytest.fixture()
def graph(tmp_path: Path) -> MikadoGraph:
    return MikadoGraph(tmp_path / "bridge.db")


class TestApplyBatchesToGraph:
    def test_linear_chain_creates_goal_root_and_batches(
        self, graph: MikadoGraph,
    ) -> None:
        manifest = _manifest(
            _change("a", "src/a.py", "Add module A"),
            _change("b", "src/b.py", "Add module B"),
            _change("c", "src/c.py", "Add module C"),
        )
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("a",), depends_on=()),
                Batch(index=1, change_ids=("b",), depends_on=(0,)),
                Batch(index=2, change_ids=("c",), depends_on=(1,)),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        # 4 nodes: goal root + 3 batches
        assert len(created) == 4
        all_nodes = graph.get_all_nodes()
        assert len(all_nodes) == 4

        # First node is goal root
        goal_root = graph.get_node(created[0])
        assert goal_root is not None
        assert goal_root.description == "bridge test goal summary"

        # Batch nodes in order
        batch_nodes = [graph.get_node(nid) for nid in created[1:]]
        assert all(n is not None for n in batch_nodes)
        assert [n.batch_index for n in batch_nodes if n] == [0, 1, 2]

        # Batch 0 attaches to goal root
        goal_children = graph.get_children(created[0])
        assert [n.id for n in goal_children] == [created[1]]

        # Linear chain edges
        middle_prereqs = graph.get_children(created[2])
        assert [n.id for n in middle_prereqs] == [created[1]]
        last_prereqs = graph.get_children(created[3])
        assert [n.id for n in last_prereqs] == [created[2]]

    def test_linear_chain_batch_descriptions_contain_stacked_intents(
        self, graph: MikadoGraph,
    ) -> None:
        manifest = _manifest(
            _change("a", "src/a.py", "Add module A"),
            _change("b", "src/b.py", "Add module B"),
        )
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("a", "b"), depends_on=()),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        batch_node = graph.get_node(created[1])
        assert batch_node is not None
        assert "1. Add module A" in batch_node.description
        assert "2. Add module B" in batch_node.description

    def test_diamond_root_batches_attach_to_goal_root(
        self, graph: MikadoGraph,
    ) -> None:
        manifest = _manifest(
            _change("root", "r.py", "Root change"),
            _change("left", "l.py", "Left change"),
            _change("right", "r2.py", "Right change"),
            _change("join", "j.py", "Join change"),
        )
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("root",), depends_on=()),
                Batch(index=1, change_ids=("left",), depends_on=(0,)),
                Batch(index=2, change_ids=("right",), depends_on=(0,)),
                Batch(index=3, change_ids=("join",), depends_on=(1, 2)),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        # 5 nodes: goal root + 4 batches
        assert len(created) == 5

        goal_root_id = created[0]
        batch_ids = created[1:]

        # Only root batch (index=0) attaches to goal root
        goal_children = graph.get_children(goal_root_id)
        assert [n.id for n in goal_children] == [batch_ids[0]]

        # Join node depends on left and right
        join_prereqs = graph.get_children(batch_ids[3])
        assert sorted(n.id for n in join_prereqs) == sorted(
            [batch_ids[1], batch_ids[2]],
        )

    def test_oversized_batch_marks_node_flag(self, graph: MikadoGraph) -> None:
        manifest = _manifest(_change("big", "huge.py", "Big oversized change"))
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("big",), depends_on=(), oversized=True),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        # created[0] = goal root, created[1] = batch node
        batch_node = graph.get_node(created[1])
        assert batch_node is not None
        assert batch_node.oversized is True
        assert batch_node.batch_index == 0

    def test_resume_with_parent_id_no_goal_root(
        self, graph: MikadoGraph,
    ) -> None:
        existing = graph.add_node("top goal")
        manifest = _manifest(
            _change("a", "a.py", "Change A"),
            _change("b", "b.py", "Change B"),
        )
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("a",), depends_on=()),
                Batch(index=1, change_ids=("b",), depends_on=(0,)),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(
            graph, plan, manifest, parent_id=existing.id,
        )

        # No goal root created — only 2 batch nodes returned
        assert len(created) == 2

        # Root batch attaches to existing node
        existing_prereqs = graph.get_children(existing.id)
        assert [n.id for n in existing_prereqs] == [created[0]]

        # Non-root batch depends on root batch
        non_root_prereqs = graph.get_children(created[1])
        assert [n.id for n in non_root_prereqs] == [created[0]]

    def test_empty_manifest_creates_only_goal_root(
        self, graph: MikadoGraph,
    ) -> None:
        """Empty manifest with no changes creates only the goal root node."""
        manifest = _manifest()
        plan = BatchPlan(
            batches=(),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        # Only the goal root node is created
        assert len(created) == 1
        goal_root = graph.get_node(created[0])
        assert goal_root is not None
        assert goal_root.description == "bridge test goal summary"

        latest = graph.get_latest_batch_plan()
        assert latest is not None
        assert latest["batch_count"] == 0
        assert latest["solver_status"] == "OPTIMAL"

    def test_description_stacking_two_changes_full_text(
        self, graph: MikadoGraph,
    ) -> None:
        long_desc_1 = "Implement the OAuth2 token refresh logic in auth module"
        long_desc_2 = "Update API client to use refreshed tokens for all requests"
        manifest = _manifest(
            _change("c1", "auth.py", long_desc_1),
            _change("c2", "client.py", long_desc_2),
        )
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("c1", "c2"), depends_on=()),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        batch_node = graph.get_node(created[1])
        assert batch_node is not None
        # Both descriptions appear in full, numbered
        assert f"1. {long_desc_1}" in batch_node.description
        assert f"2. {long_desc_2}" in batch_node.description

    def test_description_fallback_to_id_when_empty(
        self, graph: MikadoGraph,
    ) -> None:
        # Construct a change with empty description directly (bypass _change helper)
        change = FileChange(id="x1", path="x.py")
        manifest = _manifest(change)
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("x1",), depends_on=()),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        batch_node = graph.get_node(created[1])
        assert batch_node is not None
        # Fallback: id appears when description is empty
        assert "x1" in batch_node.description

    def test_missing_goal_raises_when_no_parent(self, graph: MikadoGraph) -> None:
        manifest = PlanChangeManifest(
            manifest_version=MANIFEST_VERSION,
            goal="",
            goal_summary="",
            spec_path=None,
            changes=(),
            new_relationships=(),
        )
        plan = BatchPlan(batches=(), spread_report=(), solver_status="OPTIMAL")

        with pytest.raises(ValueError, match="goal"):
            apply_batches_to_graph(graph, plan, manifest)

    def test_file_ownership_is_union_of_changes(
        self, graph: MikadoGraph,
    ) -> None:
        manifest = _manifest(
            _change("a", "src/one.py", "Change one"),
            _change("b", "src/two.py", "Change two"),
        )
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("a", "b"), depends_on=()),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        files = graph.get_file_ownership(created[1])
        assert set(files) == {"src/one.py", "src/two.py"}

    def test_end_to_end_persistence_reopens_db(
        self, tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "persist.db"
        graph = MikadoGraph(db_path)
        manifest = _manifest(
            _change("a", "a.py", "Change A"),
            _change("b", "b.py", "Change B oversized"),
        )
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("a",), depends_on=()),
                Batch(index=1, change_ids=("b",), depends_on=(0,), oversized=True),
            ),
            spread_report=(
                SymbolSpread(
                    symbol=SymbolRef(name="Foo", file="a.py"), spread=2,
                ),
            ),
            solver_status="FEASIBLE",
        )

        created = apply_batches_to_graph(graph, plan, manifest)
        graph.close()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, oversized, batch_index FROM nodes ORDER BY id",
        ).fetchall()
        # created[0] = goal root (no batch_index), created[1:] = batches
        assert [r["id"] for r in rows] == created
        # goal root: oversized=0, batch_index=None
        assert rows[0]["oversized"] == 0
        assert rows[0]["batch_index"] is None
        # batch nodes
        assert [r["oversized"] for r in rows[1:]] == [0, 1]
        assert [r["batch_index"] for r in rows[1:]] == [0, 1]

        plan_rows = conn.execute(
            "SELECT solver_status, batch_count, oversized_count, max_spread "
            "FROM batch_plans",
        ).fetchall()
        assert len(plan_rows) == 1
        only = plan_rows[0]
        assert only["solver_status"] == "FEASIBLE"
        assert only["batch_count"] == 2
        assert only["oversized_count"] == 1
        assert only["max_spread"] == 2
        conn.close()


class TestRunBatching:
    def test_forwards_manifest_to_plan_batches(self, tmp_path: Path) -> None:
        manifest = _manifest(
            _change("a", "a.py", "Change A"),
            _change("b", "b.py", "Change B"),
        )
        plan = run_batching(manifest, crg=None, root=tmp_path, budget=70_000)
        assert plan.solver_status in {"OPTIMAL", "FEASIBLE"}
        assert len(plan.batches) >= 1

    def test_empty_manifest_returns_empty_plan(self, tmp_path: Path) -> None:
        plan = run_batching(_manifest(), crg=None, root=tmp_path)
        assert plan.batches == ()
        assert plan.solver_status == "OPTIMAL"
