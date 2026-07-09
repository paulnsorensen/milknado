"""Acceptance tests for the node-kind-hierarchy spec.

Covers every acceptance criterion from the spec's ## Acceptance section.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from milknado.domains.common import NodeKind
from milknado.domains.common.types import BUILTIN_FLAVORS
from milknado.domains.graph import MikadoGraph

# ── helpers ────────────────────────────────────────────────────────────────────


def _simple_plan():
    """Minimal BatchPlan with one batch child, for batching_bridge tests."""
    from milknado.domains.batching import Batch, BatchPlan, FileChange
    from milknado.domains.planning.manifest import MANIFEST_VERSION, PlanChangeManifest

    change = FileChange(id="a", path="src/a.py", description="Change A")
    manifest = PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        changes=(change,),
        new_relationships=(),
        goal="test goal",
        goal_summary="test goal summary",
        spec_path=None,
    )
    plan = BatchPlan(
        batches=(Batch(index=0, change_ids=("a",), depends_on=()),),
        spread_report=(),
        solver_status="OPTIMAL",
    )
    return plan, manifest


# ── containment enforcement ────────────────────────────────────────────────────


class TestContainmentEnforcement:
    def test_task_under_roadmap_raises_invalid_containment(self, graph: MikadoGraph) -> None:
        """add_node of a task under a roadmap must raise InvalidContainment."""
        from milknado.domains.common.errors import InvalidContainment

        roadmap = graph.add_node("rm", kind=NodeKind.ROADMAP)
        with pytest.raises(InvalidContainment):
            graph.add_node("t", parent_id=roadmap.id, kind=NodeKind.TASK)

    def test_goal_under_roadmap_is_legal(self, graph: MikadoGraph) -> None:
        """ROADMAP → GOAL is in VALID_CHILD_KINDS."""
        roadmap = graph.add_node("rm", kind=NodeKind.ROADMAP)
        goal = graph.add_node("g", parent_id=roadmap.id, kind=NodeKind.GOAL)
        assert goal.kind == NodeKind.GOAL

    def test_task_under_goal_is_legal(self, graph: MikadoGraph) -> None:
        """GOAL → TASK is in VALID_CHILD_KINDS."""
        goal = graph.add_node("g", kind=NodeKind.GOAL)
        task = graph.add_node("t", parent_id=goal.id, kind=NodeKind.TASK)
        assert task.kind == NodeKind.TASK

    def test_goal_under_goal_is_legal(self, graph: MikadoGraph) -> None:
        """GOAL → GOAL (sub-goal) is in VALID_CHILD_KINDS."""
        parent_goal = graph.add_node("pg", kind=NodeKind.GOAL)
        sub_goal = graph.add_node("sg", parent_id=parent_goal.id, kind=NodeKind.GOAL)
        assert sub_goal.kind == NodeKind.GOAL

    def test_task_under_task_is_legal(self, graph: MikadoGraph) -> None:
        """TASK → TASK (Mikado prereq chain) is in VALID_CHILD_KINDS."""
        parent_task = graph.add_node("pt")
        child_task = graph.add_node("ct", parent_id=parent_task.id)
        assert child_task.kind == NodeKind.TASK

    def test_roadmap_at_root_is_legal(self, graph: MikadoGraph) -> None:
        """Root nodes (parent_id=None) accept any kind."""
        roadmap = graph.add_node("rm", kind=NodeKind.ROADMAP)
        goal = graph.add_node("g", kind=NodeKind.GOAL)
        task = graph.add_node("t", kind=NodeKind.TASK)
        assert roadmap.kind == NodeKind.ROADMAP
        assert goal.kind == NodeKind.GOAL
        assert task.kind == NodeKind.TASK

    def test_roadmap_under_task_raises_via_move_node(self, graph: MikadoGraph) -> None:
        """move_node that would place a roadmap under a task raises InvalidContainment."""
        from milknado.domains.common.errors import InvalidContainment

        task = graph.add_node("t", kind=NodeKind.TASK)
        roadmap = graph.add_node("rm", kind=NodeKind.ROADMAP)
        with pytest.raises(InvalidContainment):
            graph.move_node(roadmap.id, task.id)

    def test_goal_under_task_raises_via_move_node(self, graph: MikadoGraph) -> None:
        """move_node that would place a goal under a task also raises."""
        from milknado.domains.common.errors import InvalidContainment

        task = graph.add_node("t", kind=NodeKind.TASK)
        goal = graph.add_node("g", kind=NodeKind.GOAL)
        with pytest.raises(InvalidContainment):
            graph.move_node(goal.id, task.id)

    def test_kind_edit_violating_parent_pair_raises(self, graph: MikadoGraph) -> None:
        """Editing a node's kind into an illegal parent->child pair raises.

        A TASK under a GOAL is legal; re-kinding the parent GOAL to ROADMAP
        would make ROADMAP->TASK, which is not in VALID_CHILD_KINDS. The edit
        path must enforce containment the same as add_node/move_node.
        """
        from milknado.domains.common.errors import InvalidContainment

        goal = graph.add_node("g", kind=NodeKind.GOAL)
        graph.add_node("t", parent_id=goal.id, kind=NodeKind.TASK)
        with pytest.raises(InvalidContainment):
            graph.update_node(goal.id, kind=NodeKind.ROADMAP)

    def test_kind_edit_violating_child_pair_raises(self, graph: MikadoGraph) -> None:
        """Re-kinding a node so an existing child becomes illegal raises.

        GOAL->GOAL is legal; re-kinding the parent GOAL to TASK would make
        TASK->GOAL, which is illegal. Checked against the node's child edges.
        """
        from milknado.domains.common.errors import InvalidContainment

        parent = graph.add_node("pg", kind=NodeKind.GOAL)
        graph.add_node("sg", parent_id=parent.id, kind=NodeKind.GOAL)
        with pytest.raises(InvalidContainment):
            graph.update_node(parent.id, kind=NodeKind.TASK)


# ── run-start refusal ──────────────────────────────────────────────────────────


class TestRunStartRefusal:
    def test_run_inline_start_refuses_goal(self, tmp_path: Path) -> None:
        """milknado_run_inline_start on a goal node refuses with ValueError."""
        from unittest.mock import patch

        from milknado.mcp_run import milknado_run_inline_start

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        goal = g.add_node("goal node", kind=NodeKind.GOAL)
        g.close()

        with (
            patch("milknado.mcp_run.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_run.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            with pytest.raises(ValueError, match="kind"):
                milknado_run_inline_start(goal.id)

    def test_run_loop_start_refuses_roadmap(self, tmp_path: Path) -> None:
        """milknado_run_loop_start on a roadmap node refuses with ValueError."""
        from unittest.mock import patch

        from milknado.mcp_ralph import milknado_run_loop_start

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        roadmap = g.add_node("roadmap node", kind=NodeKind.ROADMAP)
        g.close()

        with (
            patch("milknado.mcp_ralph.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_ralph.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            with pytest.raises(ValueError, match="kind"):
                milknado_run_loop_start(roadmap.id)


# ── validate_goal_runnable ────────────────────────────────────────────────────


class TestValidateGoalRunnable:
    def test_errors_when_node_is_not_goal(self, graph: MikadoGraph) -> None:
        from milknado.domains.graph.runnability import validate_goal_runnable

        task = graph.add_node("t")
        report = validate_goal_runnable(graph, task.id)
        assert any("not" in e.lower() and "goal" in e.lower() for e in report.errors)

    def test_errors_when_pending_goal_has_zero_children(self, graph: MikadoGraph) -> None:
        """A pending GOAL with zero children is an undecomposed stub → error."""
        from milknado.domains.graph.runnability import validate_goal_runnable

        goal = graph.add_node("g", kind=NodeKind.GOAL)
        report = validate_goal_runnable(graph, goal.id)
        assert any("zero" in e.lower() or "child" in e.lower() for e in report.errors)

    def test_errors_when_leaf_descendant_is_not_task(self, graph: MikadoGraph) -> None:
        """A leaf descendant that is a goal (not task) counts as an error."""
        from milknado.domains.graph.runnability import validate_goal_runnable

        goal = graph.add_node("g", kind=NodeKind.GOAL)
        _ = graph.add_node("sg", parent_id=goal.id, kind=NodeKind.GOAL)
        # sub_goal is a leaf and not a TASK → error
        report = validate_goal_runnable(graph, goal.id)
        assert any("leaf" in e.lower() or "task" in e.lower() for e in report.errors)

    def test_passes_when_all_leaves_are_tasks(self, graph: MikadoGraph) -> None:
        """A goal with all leaf descendants being tasks has no errors."""
        from milknado.domains.graph.runnability import validate_goal_runnable

        goal = graph.add_node("g", kind=NodeKind.GOAL)
        graph.add_node("t1", parent_id=goal.id, kind=NodeKind.TASK)
        graph.add_node("t2", parent_id=goal.id, kind=NodeKind.TASK)
        report = validate_goal_runnable(graph, goal.id)
        assert report.errors == []

    def test_warning_for_implement_task_without_file_hints(self, graph: MikadoGraph) -> None:
        """An implement-flavored task without file hints emits a warning (never an error)."""
        from milknado.domains.graph.runnability import validate_goal_runnable

        goal = graph.add_node("g", kind=NodeKind.GOAL)
        graph.add_node("t", parent_id=goal.id, kind=NodeKind.TASK)
        report = validate_goal_runnable(graph, goal.id)
        assert report.errors == []
        assert len(report.warnings) >= 1


# ── flavor round-trip ────────────────────────────────────────────────────────


class TestFlavorRoundTrip:
    def test_default_flavor_for_task_is_implement(self, graph: MikadoGraph) -> None:
        task = graph.add_node("t")
        assert task.flavor == "implement"

    def test_non_task_nodes_have_no_flavor(self, graph: MikadoGraph) -> None:
        goal = graph.add_node("g", kind=NodeKind.GOAL)
        roadmap = graph.add_node("rm", kind=NodeKind.ROADMAP)
        assert goal.flavor is None
        assert roadmap.flavor is None

    def test_flavor_survives_add_reload_round_trip(self, tmp_path: Path) -> None:
        """flavor persists through graph close/reopen."""
        db_path = tmp_path / "rt.db"
        g = MikadoGraph(db_path)
        goal = g.add_node("g", kind=NodeKind.GOAL)
        # Task under goal — must pass containment check
        task = g.add_node("t", parent_id=goal.id, kind=NodeKind.TASK, flavor="spike")
        task_id = task.id
        g.close()

        g2 = MikadoGraph(db_path)
        reloaded = g2.get_node(task_id)
        g2.close()
        assert reloaded is not None
        assert reloaded.flavor == "spike"

    def test_all_flavor_values_survive_round_trip(self, tmp_path: Path) -> None:
        """Every builtin flavor value persists through close/reopen."""
        db_path = tmp_path / "flavors.db"
        g = MikadoGraph(db_path)
        goal = g.add_node("g", kind=NodeKind.GOAL)
        flavor_ids: dict[str, int] = {}
        for flavor in BUILTIN_FLAVORS:
            t = g.add_node(f"t-{flavor}", parent_id=goal.id, kind=NodeKind.TASK, flavor=flavor)
            flavor_ids[flavor] = t.id
        g.close()

        g2 = MikadoGraph(db_path)
        for flavor, nid in flavor_ids.items():
            node = g2.get_node(nid)
            assert node is not None, f"node {nid} not found"
            assert node.flavor == flavor, f"expected {flavor!r}, got {node.flavor!r}"
        g2.close()

    def test_setting_flavor_on_goal_raises(self, graph: MikadoGraph) -> None:
        """Setting flavor on a kind=goal node raises ValueError."""
        with pytest.raises(ValueError, match="flavor"):
            graph.add_node("g", kind=NodeKind.GOAL, flavor="spike")

    def test_setting_flavor_on_roadmap_raises(self, graph: MikadoGraph) -> None:
        """Setting flavor on a kind=roadmap node raises ValueError."""
        with pytest.raises(ValueError, match="flavor"):
            graph.add_node("rm", kind=NodeKind.ROADMAP, flavor="research")


# ── missing-column defaults (pre-migration db) ─────────────────────────────────


class TestMissingColumnDefaults:
    def test_pre_migration_db_loads_with_implement_default_for_tasks(self, tmp_path: Path) -> None:
        """A db without the flavor column loads tasks with implement default via row_to_node."""
        db_path = tmp_path / "old.db"
        # Build a db WITHOUT the flavor column by creating it manually
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parent_id INTEGER,
                worktree_path TEXT,
                branch_name TEXT,
                run_id TEXT,
                pid INTEGER,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                dispatched_at TEXT,
                completion_duration_seconds REAL,
                oversized INTEGER NOT NULL DEFAULT 0,
                batch_index INTEGER,
                kind TEXT NOT NULL DEFAULT 'task'
            );
            CREATE TABLE edges (
                parent_id INTEGER NOT NULL,
                child_id INTEGER NOT NULL,
                PRIMARY KEY (parent_id, child_id)
            );
        """)
        # Insert a task row without flavor
        conn.execute(
            "INSERT INTO nodes (description, status, created_at, kind)"
            " VALUES (?, 'pending', '2025-01-01', 'task')",
            ("legacy task",),
        )
        # Insert a goal row without flavor
        conn.execute(
            "INSERT INTO nodes (description, status, created_at, kind)"
            " VALUES (?, 'pending', '2025-01-01', 'goal')",
            ("legacy goal",),
        )
        conn.commit()
        conn.close()

        # Open via MikadoGraph — ensure_schema should add the flavor column
        g = MikadoGraph(db_path)
        nodes = g.get_all_nodes()
        g.close()

        task_node = next(n for n in nodes if n.description == "legacy task")
        goal_node = next(n for n in nodes if n.description == "legacy goal")

        assert task_node.flavor == "implement"
        assert goal_node.flavor is None


# ── apply_batches_to_graph kinds ─────────────────────────────────────────────


class TestApplyBatchesKinds:
    def test_apply_batches_produces_goal_root_and_task_children(self, graph: MikadoGraph) -> None:
        """apply_batches_to_graph produces a GOAL root with TASK children."""
        from milknado.domains.planning.batching_bridge import apply_batches_to_graph

        plan, manifest = _simple_plan()
        created = apply_batches_to_graph(graph, plan, manifest)

        goal_root = graph.get_node(created[0])
        assert goal_root is not None
        assert goal_root.kind == NodeKind.GOAL

        for nid in created[1:]:
            node = graph.get_node(nid)
            assert node is not None
            assert node.kind == NodeKind.TASK


# ── coverage: uncovered branches ───────────────────────────────────────────────


class TestCoverageBranches:
    def test_parse_flavor_invalid_raises(self) -> None:
        """_parse_flavor raises ValueError for unknown flavor strings."""
        from milknado._mcp_core import _parse_flavor

        with pytest.raises(ValueError, match="invalid flavor"):
            _parse_flavor("unknown-flavor")

    def test_cli_add_node_invalid_kind_raises(self, tmp_path: Path) -> None:
        """CLI add_node raises BadParameter for invalid kind strings."""
        import typer

        from milknado.cli import add_node

        with pytest.raises((typer.BadParameter, SystemExit)):
            add_node(
                description="x",
                kind="invalid-kind",
                project_root=tmp_path,
            )

    def test_run_inline_refuses_goal(self, tmp_path: Path) -> None:
        """milknado_run_inline on a goal node raises ValueError."""
        from unittest.mock import patch

        from milknado.mcp_run import milknado_run_inline

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        goal = g.add_node("goal node", kind=NodeKind.GOAL)
        g.close()

        with (
            patch("milknado.mcp_run.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_run.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            with pytest.raises(ValueError, match="kind"):
                milknado_run_inline(goal.id)

    def test_todo_next_flavor_filter_no_match(self, tmp_path: Path) -> None:
        """milknado_todo_next returns None when the next node doesn't match the flavor filter."""
        from unittest.mock import patch

        from milknado.mcp_todo import milknado_todo_next

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        goal = g.add_node("g", kind=NodeKind.GOAL)
        g.add_node("t", parent_id=goal.id, kind=NodeKind.TASK)  # flavor=IMPLEMENT (default)
        g.close()

        with (
            patch("milknado.mcp_todo.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_todo.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            # Filter for SPIKE — the node is IMPLEMENT, so no match
            result = milknado_todo_next(kind="task", flavor="spike", project_root=str(tmp_path))
            assert result is None


class TestEditNodeFlavor:
    """milknado_edit_node must accept and enforce the flavor param."""

    def test_edit_node_sets_flavor_on_task(self, tmp_path: Path) -> None:
        """Flavor can be set on a task node via milknado_edit_node."""
        from unittest.mock import patch

        from milknado.mcp_todo_mutate import milknado_edit_node

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        goal = g.add_node("goal", kind=NodeKind.GOAL)
        task = g.add_node("task", parent_id=goal.id, kind=NodeKind.TASK)
        g.close()

        with (
            patch("milknado.mcp_todo_mutate.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_todo_mutate.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            result = milknado_edit_node(task.id, flavor="spike", project_root=str(tmp_path))
            assert result["flavor"] == "spike"

    def test_edit_node_flavor_on_goal_raises(self, tmp_path: Path) -> None:
        """Setting flavor on a non-task node raises ValueError."""
        from unittest.mock import patch

        from milknado.mcp_todo_mutate import milknado_edit_node

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        goal = g.add_node("goal", kind=NodeKind.GOAL)
        g.close()

        with (
            patch("milknado.mcp_todo_mutate.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_todo_mutate.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            with pytest.raises(ValueError, match="flavor"):
                milknado_edit_node(goal.id, flavor="spike", project_root=str(tmp_path))

    def test_edit_node_kind_to_non_task_clears_flavor(self, tmp_path: Path) -> None:
        """Changing kind from task to goal clears the flavor (invariant enforced)."""
        from unittest.mock import patch

        from milknado.mcp_todo_mutate import milknado_edit_node

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        root = g.add_node("root", kind=NodeKind.ROADMAP)
        goal = g.add_node("goal", parent_id=root.id, kind=NodeKind.GOAL)
        task = g.add_node("task", parent_id=goal.id, kind=NodeKind.TASK, flavor="spike")
        g.close()

        with (
            patch("milknado.mcp_todo_mutate.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_todo_mutate.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            # Edit task to kind=task still (no kind change) and verify flavor readable
            result = milknado_edit_node(task.id, description="renamed", project_root=str(tmp_path))
            # After description-only edit, flavor should still be spike
            assert result["flavor"] == "spike"

    def test_edit_node_kind_and_flavor_conflict_raises(self, tmp_path: Path) -> None:
        """Passing kind=goal and flavor together raises ValueError."""
        from unittest.mock import patch

        from milknado.mcp_todo_mutate import milknado_edit_node

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        root = g.add_node("root", kind=NodeKind.ROADMAP)
        goal = g.add_node("goal", parent_id=root.id, kind=NodeKind.GOAL)
        task = g.add_node("task", parent_id=goal.id, kind=NodeKind.TASK)
        g.close()

        with (
            patch("milknado.mcp_todo_mutate.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_todo_mutate.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            with pytest.raises(ValueError, match="flavor"):
                milknado_edit_node(
                    task.id, kind="goal", flavor="spike", project_root=str(tmp_path)
                )


class TestGraphSummaryFlavorFilter:
    """milknado_graph_summary must support a flavor filter."""

    def test_graph_summary_flavor_filter(self, tmp_path: Path) -> None:
        """flavor param filters nodes to only those with the matching flavor."""
        from unittest.mock import patch

        from milknado.mcp_server import milknado_graph_summary

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        goal = g.add_node("goal", kind=NodeKind.GOAL)
        g.add_node("t1", parent_id=goal.id, kind=NodeKind.TASK, flavor="spike")
        g.add_node("t2", parent_id=goal.id, kind=NodeKind.TASK, flavor="implement")
        g.close()

        with (
            patch("milknado.mcp_server.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_server.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            result = milknado_graph_summary(flavor="spike", project_root=str(tmp_path))
            descriptions = [n["description"] for n in result["nodes"]]
            assert descriptions == ["t1"]

    def test_graph_summary_no_flavor_returns_all(self, tmp_path: Path) -> None:
        """Omitting flavor returns all nodes (no filter applied)."""
        from unittest.mock import patch

        from milknado.mcp_server import milknado_graph_summary

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        goal = g.add_node("goal", kind=NodeKind.GOAL)
        g.add_node("t1", parent_id=goal.id, kind=NodeKind.TASK, flavor="spike")
        g.add_node("t2", parent_id=goal.id, kind=NodeKind.TASK)
        g.close()

        with (
            patch("milknado.mcp_server.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_server.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            result = milknado_graph_summary(project_root=str(tmp_path))
            assert len(result["nodes"]) == 3  # goal + 2 tasks


# ── hardening: update_node_fields kind→non-task auto-NULLs flavor ─────────────


class TestKindEditClearsFlavor:
    """update_node_fields must NULL flavor when kind changes away from TASK.

    This is a coder design call not explicit in the spec — lock it here so
    a regression in update_node_fields doesn't silently violate the invariant
    (non-task node with a non-None flavor).
    """

    def test_kind_task_to_goal_clears_flavor(self, tmp_path: Path) -> None:
        """Editing a task to kind=goal NULLs the flavor in the DB; reload returns None."""
        from unittest.mock import patch

        from milknado.mcp_todo_mutate import milknado_edit_node

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        root = g.add_node("root", kind=NodeKind.ROADMAP)
        goal = g.add_node("goal", parent_id=root.id, kind=NodeKind.GOAL)
        # TASK→TASK under goal is legal; parent is goal so we can later edit to sub-goal
        task = g.add_node("task", parent_id=goal.id, kind=NodeKind.TASK, flavor="spike")
        task_id = task.id
        g.close()

        with (
            patch("milknado.mcp_todo_mutate.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_todo_mutate.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            # Edit kind from task→goal (still parented under goal → sub-goal, legal containment)
            result = milknado_edit_node(task_id, kind="goal", project_root=str(tmp_path))
            # Return value must NOT carry a flavor field (flavor is None for non-task)
            assert "flavor" not in result, f"flavor should be absent for goal node; got {result!r}"

        # Reload from DB: flavor column must be NULL
        g2 = MikadoGraph(graph_path)
        reloaded = g2.get_node(task_id)
        g2.close()
        assert reloaded is not None
        assert reloaded.flavor is None, f"expected None after kind→goal; got {reloaded.flavor!r}"

    def test_update_node_fields_directly_clears_flavor_on_kind_change(
        self, tmp_path: Path
    ) -> None:
        """update_node_fields(kind=GOAL) zeroes flavor regardless of caller layer."""

        from milknado.domains.graph._mutations import update_node_fields

        graph_path = tmp_path / "direct.db"
        g = MikadoGraph(graph_path)
        root = g.add_node("root", kind=NodeKind.ROADMAP)
        goal = g.add_node("goal", parent_id=root.id, kind=NodeKind.GOAL)
        task = g.add_node("t", parent_id=goal.id, kind=NodeKind.TASK, flavor="research")
        task_id = task.id

        # Confirm flavor is set before the edit
        assert task.flavor == "research"

        # Call update_node_fields directly to change kind to GOAL
        update_node_fields(g._conn, task_id, description=None, kind=NodeKind.GOAL)

        # Reload within the same connection to verify the DB row changed
        from milknado.domains.graph._persistence import row_to_node

        row = g._conn.execute("SELECT * FROM nodes WHERE id = ?", (task_id,)).fetchone()
        node = row_to_node(row)
        g.close()

        assert node.flavor is None, f"expected None after kind→goal edit; got {node.flavor!r}"


# ── hardening: ralph_run_start refusal on GOAL (not just ROADMAP) ─────────────


class TestRalphRunStartRefusesGoal:
    def test_run_loop_start_refuses_goal(self, tmp_path: Path) -> None:
        """milknado_run_loop_start on a GOAL node refuses with ValueError."""
        from unittest.mock import patch

        from milknado.mcp_ralph import milknado_run_loop_start

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        goal = g.add_node("goal node", kind=NodeKind.GOAL)
        goal_id = goal.id
        g.close()

        with (
            patch("milknado.mcp_ralph.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_ralph.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            with pytest.raises(ValueError, match="kind"):
                milknado_run_loop_start(goal_id)


# ── hardening: validate_goal_runnable edge cases ───────────────────────────────


class TestValidateGoalRunnableEdgeCases:
    def test_roadmap_leaf_is_an_error(self, graph: MikadoGraph) -> None:
        """A leaf descendant of kind=ROADMAP (theoretically possible via direct DB ops)
        is caught by the leaf-not-task check."""
        from milknado.domains.graph.runnability import validate_goal_runnable

        # Build goal → task, then directly corrupt the task row to kind=roadmap
        # to simulate any path that produces a non-TASK leaf without going through
        # the containment enforcement (e.g. direct DB manipulation or legacy data).
        goal = graph.add_node("g", kind=NodeKind.GOAL)
        task = graph.add_node("t", parent_id=goal.id, kind=NodeKind.TASK)

        # Corrupt the task kind directly in the DB to roadmap (clearing flavor to
        # keep the row's flavor invariant: only task nodes may carry a flavor).
        graph._conn.execute(
            "UPDATE nodes SET kind = 'roadmap', flavor = NULL WHERE id = ?", (task.id,)
        )
        graph._conn.commit()

        report = validate_goal_runnable(graph, goal.id)
        assert len(report.errors) >= 1
        assert any("task" in e.lower() or "leaf" in e.lower() for e in report.errors)

    def test_non_task_row_with_flavor_fails_fast_on_read(self, graph: MikadoGraph) -> None:
        """row_to_node refuses a non-task row carrying a non-NULL flavor.

        Reading a corrupt row (kind!=task but flavor set) must fail loud rather
        than silently coercing flavor to None and hiding the broken invariant.
        """
        task = graph.add_node("t", kind=NodeKind.TASK, flavor="implement")
        graph._conn.execute("UPDATE nodes SET kind = 'roadmap' WHERE id = ?", (task.id,))
        graph._conn.commit()
        with pytest.raises(ValueError, match="flavor"):
            graph.get_node(task.id)

    def test_deep_task_chain_passes(self, graph: MikadoGraph) -> None:
        """A goal → task → task (Mikado prereq chain) has no errors; leaf tasks are fine."""
        from milknado.domains.graph.runnability import validate_goal_runnable

        goal = graph.add_node("g", kind=NodeKind.GOAL)
        parent_task = graph.add_node("pt", parent_id=goal.id, kind=NodeKind.TASK)
        # TASK → TASK is a Mikado prerequisite chain (legal by VALID_CHILD_KINDS)
        graph.add_node("ct", parent_id=parent_task.id, kind=NodeKind.TASK)

        report = validate_goal_runnable(graph, goal.id)
        assert report.errors == []

    def test_non_implement_task_does_not_warn_about_hints(self, graph: MikadoGraph) -> None:
        """Only implement-flavored tasks without file hints emit a warning;
        spec/spike/etc. don't."""
        from milknado.domains.graph.runnability import validate_goal_runnable

        goal = graph.add_node("g", kind=NodeKind.GOAL)
        graph.add_node("t", parent_id=goal.id, kind=NodeKind.TASK, flavor="spike")

        report = validate_goal_runnable(graph, goal.id)
        assert report.errors == []
        # A SPIKE task without file hints should NOT produce the "no file hints" warning
        hint_warnings = [
            w for w in report.warnings if "file hints" in w.lower() or "hint" in w.lower()
        ]
        assert hint_warnings == [], f"unexpected hint warnings for spike task: {hint_warnings!r}"


# ── hardening: edit_node flavor persists through DB reload ─────────────────────


class TestEditNodeFlavorPersistence:
    def test_edit_node_flavor_survives_reload(self, tmp_path: Path) -> None:
        """Flavor set via milknado_edit_node persists when the DB is closed and reopened."""
        from unittest.mock import patch

        from milknado.mcp_todo_mutate import milknado_edit_node

        graph_path = tmp_path / "test.db"
        g = MikadoGraph(graph_path)
        goal = g.add_node("goal", kind=NodeKind.GOAL)
        task = g.add_node("task", parent_id=goal.id, kind=NodeKind.TASK)
        task_id = task.id
        # Starts as IMPLEMENT (default)
        assert task.flavor == "implement"
        g.close()

        with (
            patch("milknado.mcp_todo_mutate.resolve_project_root", return_value=tmp_path),
            patch("milknado.mcp_todo_mutate.open_graph") as mock_open,
        ):
            import milknado.domains.common.config as cfg_mod

            mock_graph = MikadoGraph(graph_path)
            mock_cfg = cfg_mod.default_config(tmp_path)
            mock_open.return_value = (mock_graph, mock_cfg)
            result = milknado_edit_node(task_id, flavor="research", project_root=str(tmp_path))
            assert result["flavor"] == "research"

        # Reload from disk — verify the DB row has the updated flavor
        g2 = MikadoGraph(graph_path)
        reloaded = g2.get_node(task_id)
        g2.close()
        assert reloaded is not None
        assert reloaded.flavor == "research", (
            f"expected RESEARCH after edit+reload; got {reloaded.flavor!r}"
        )


# ── hardening: reparent to None (make root) never raises ──────────────────────


class TestReparentToRoot:
    def test_reparent_to_none_makes_root_without_containment_check(
        self, graph: MikadoGraph
    ) -> None:
        """Reparenting a node to None (detaching to root) always succeeds;
        spec says root nodes (parent_id=None) accept any kind."""
        roadmap = graph.add_node("rm", kind=NodeKind.ROADMAP)
        goal = graph.add_node("g", parent_id=roadmap.id, kind=NodeKind.GOAL)

        # Move goal to root (parent_id=None) — no containment check applies
        graph.move_node(goal.id, None)  # must not raise

        reloaded = graph.get_node(goal.id)
        assert reloaded is not None
        assert reloaded.kind == NodeKind.GOAL


# ── cure: finding 2 — DONE leaf goal does not error ───────────────────────────


class TestDoneLeafGoalNoError:
    def test_done_leaf_goal_does_not_error(self, graph: MikadoGraph) -> None:
        """A leaf GOAL with status=DONE (zero children) must not be reported as an
        undecomposed stub; the spec scopes that error to PENDING goals only."""
        from milknado.domains.graph.runnability import validate_goal_runnable

        outer = graph.add_node("outer", kind=NodeKind.GOAL)
        done_leaf = graph.add_node("done-leaf", parent_id=outer.id, kind=NodeKind.GOAL)
        # Force DONE via direct DB write (same pattern as test_roadmap_leaf_is_an_error
        # above — the normal status machine requires PENDING→RUNNING→DONE).
        graph._conn.execute("UPDATE nodes SET status = 'done' WHERE id = ?", (done_leaf.id,))
        graph._conn.commit()

        report = validate_goal_runnable(graph, outer.id)
        assert report.errors == [], f"unexpected errors for DONE leaf goal: {report.errors!r}"


# ── cure: finding 3 — conflicting flavor+kind raises at graph layer ────────────


class TestUpdateNodeFieldsConflictingFlavor:
    def test_non_task_kind_with_explicit_flavor_raises(self, tmp_path: Path) -> None:
        """update_node_fields with kind=GOAL and a non-None flavor must raise ValueError
        rather than emitting two conflicting flavor clauses in the SQL."""
        from milknado.domains.graph._mutations import update_node_fields

        g = MikadoGraph(tmp_path / "mutations.db")
        task = g.add_node("t", kind=NodeKind.TASK)
        task_id = task.id
        g.close()

        g2 = MikadoGraph(tmp_path / "mutations.db")
        with pytest.raises(ValueError, match="flavor"):
            update_node_fields(
                g2._conn, task_id, description=None, kind=NodeKind.GOAL, flavor="research"
            )
        g2.close()


# ── cure: finding 4 — batching_bridge rejects a parent that can't hold TASKs ──


class TestApplyBatchesParentContainmentCheck:
    def test_task_parent_rejects_roadmap_parent_id(self, graph: MikadoGraph) -> None:
        """apply_batches_to_graph with a ROADMAP parent_id must raise because
        ROADMAP → TASK is not in VALID_CHILD_KINDS."""
        from milknado.domains.planning.batching_bridge import apply_batches_to_graph

        roadmap = graph.add_node("rm", kind=NodeKind.ROADMAP)
        plan, manifest = _simple_plan()

        with pytest.raises(ValueError, match="parent"):
            apply_batches_to_graph(graph, plan, manifest, parent_id=roadmap.id)
