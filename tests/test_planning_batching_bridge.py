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


def _change(cid: str, path: str, **kw: object) -> FileChange:
    kwargs: dict[str, object] = {"id": cid, "path": path}
    kwargs.update(kw)
    return FileChange(**kwargs)  # type: ignore[arg-type]


def _manifest(
    *changes: FileChange,
    new_rels: tuple[NewRelationship, ...] = (),
) -> PlanChangeManifest:
    return PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        changes=changes,
        new_relationships=new_rels,
    )


@pytest.fixture()
def graph(tmp_path: Path) -> MikadoGraph:
    return MikadoGraph(tmp_path / "bridge.db")


class TestApplyBatchesToGraph:
    def test_linear_chain_creates_nodes_in_order(
        self, graph: MikadoGraph,
    ) -> None:
        manifest = _manifest(
            _change("a", "src/a.py"),
            _change("b", "src/b.py"),
            _change("c", "src/c.py"),
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

        assert len(created) == 3
        nodes = [graph.get_node(nid) for nid in created]
        assert [n.description for n in nodes if n] == [
            "Batch 0: src/a.py",
            "Batch 1: src/b.py",
            "Batch 2: src/c.py",
        ]
        assert [n.batch_index for n in nodes if n] == [0, 1, 2]
        assert all(n is not None and not n.oversized for n in nodes)
        # dependency edges: middle node depends on first, last depends on middle
        middle_prereqs = graph.get_children(created[1])
        assert [n.id for n in middle_prereqs] == [created[0]]
        last_prereqs = graph.get_children(created[2])
        assert [n.id for n in last_prereqs] == [created[1]]

    def test_diamond_wires_both_parents(self, graph: MikadoGraph) -> None:
        manifest = _manifest(
            _change("root", "r.py"),
            _change("left", "l.py"),
            _change("right", "r2.py"),
            _change("join", "j.py"),
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

        assert len(created) == 4
        join_prereqs = graph.get_children(created[3])
        assert sorted(n.id for n in join_prereqs) == sorted(
            [created[1], created[2]],
        )

    def test_oversized_batch_marks_node_flag(self, graph: MikadoGraph) -> None:
        manifest = _manifest(_change("big", "huge.py"))
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("big",), depends_on=(), oversized=True),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        node = graph.get_node(created[0])
        assert node is not None
        assert node.oversized is True
        assert node.batch_index == 0

    def test_empty_plan_records_snapshot_and_returns_empty(
        self, graph: MikadoGraph,
    ) -> None:
        manifest = _manifest()
        plan = BatchPlan(
            batches=(),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        assert created == []
        latest = graph.get_latest_batch_plan()
        assert latest is not None
        assert latest["batch_count"] == 0
        assert latest["solver_status"] == "OPTIMAL"

    def test_resume_parent_attaches_only_roots(
        self, graph: MikadoGraph,
    ) -> None:
        existing = graph.add_node("top goal")
        manifest = _manifest(
            _change("a", "a.py"),
            _change("b", "b.py"),
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

        # Root batch becomes a prerequisite of the resume parent.
        existing_prereqs = graph.get_children(existing.id)
        assert [n.id for n in existing_prereqs] == [created[0]]
        # Non-root batch is not attached to the resume parent.
        non_root_prereqs = graph.get_children(created[1])
        assert [n.id for n in non_root_prereqs] == [created[0]]

    def test_file_ownership_is_union_of_changes(
        self, graph: MikadoGraph,
    ) -> None:
        manifest = _manifest(
            _change("a", "src/one.py"),
            _change("b", "src/two.py"),
        )
        plan = BatchPlan(
            batches=(
                Batch(index=0, change_ids=("a", "b"), depends_on=()),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        files = graph.get_file_ownership(created[0])
        assert set(files) == {"src/one.py", "src/two.py"}

    def test_description_truncates_many_files(
        self, graph: MikadoGraph,
    ) -> None:
        manifest = _manifest(
            _change("a", "a.py"),
            _change("b", "b.py"),
            _change("c", "c.py"),
            _change("d", "d.py"),
            _change("e", "e.py"),
        )
        plan = BatchPlan(
            batches=(
                Batch(
                    index=0,
                    change_ids=("a", "b", "c", "d", "e"),
                    depends_on=(),
                ),
            ),
            spread_report=(),
            solver_status="OPTIMAL",
        )

        created = apply_batches_to_graph(graph, plan, manifest)

        node = graph.get_node(created[0])
        assert node is not None
        assert node.description == "Batch 0: a.py, b.py, c.py (+2 more)"

    def test_end_to_end_persistence_reopens_db(
        self, tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "persist.db"
        graph = MikadoGraph(db_path)
        manifest = _manifest(
            _change("a", "a.py"),
            _change("b", "b.py"),
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
        assert [r["id"] for r in rows] == created
        assert [r["oversized"] for r in rows] == [0, 1]
        assert [r["batch_index"] for r in rows] == [0, 1]

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
            _change("a", "a.py"),
            _change("b", "b.py"),
        )
        plan = run_batching(manifest, crg=None, root=tmp_path, budget=70_000)
        assert plan.solver_status in {"OPTIMAL", "FEASIBLE"}
        assert len(plan.batches) >= 1

    def test_empty_manifest_returns_empty_plan(self, tmp_path: Path) -> None:
        plan = run_batching(_manifest(), crg=None, root=tmp_path)
        assert plan.batches == ()
        assert plan.solver_status == "OPTIMAL"
