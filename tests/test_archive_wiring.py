"""Wiring tests: CLI + MCP surfaces for archive lifecycle, include_archived, rebalance.

Pins the observable contracts of the wiring rows (W1-W6): the facade
delegations, the CLI commands (graph archive|unarchive, status --all,
rebalance), and the MCP tools (archive/unarchive node, include_archived on
tree/summary, set_subtree_status skip-archived, milknado_rebalance).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.common import default_config
from milknado.domains.graph import MikadoGraph
from milknado.mcp.rebalance import milknado_rebalance
from milknado.mcp.server import milknado_graph_summary
from milknado.mcp.todo import milknado_get_node, milknado_todo_tree
from milknado.mcp.todo_mutate import (
    milknado_archive_node,
    milknado_set_subtree_status,
    milknado_todo_add,
    milknado_todo_set_status,
    milknado_unarchive_node,
)

runner = CliRunner()


def _call(tool, **kwargs):
    """Invoke a FastMCP tool through its underlying Python callable."""
    return getattr(tool, "fn", tool)(**kwargs)


def _complete(graph: MikadoGraph, node_id: int) -> None:
    graph.mark_running(node_id)
    graph.mark_done(node_id)


def _seed_done_subtree(root: str) -> tuple[int, int]:
    """goal with one task, both DONE; returns (goal_id, task_id)."""
    goal = _call(milknado_todo_add, description="g", kind="goal", project_root=root)
    task = _call(
        milknado_todo_add, description="t", kind="task", parent_id=goal["id"], project_root=root
    )
    _call(milknado_todo_set_status, node_id=task["id"], status="done", project_root=root)
    _call(milknado_todo_set_status, node_id=goal["id"], status="done", project_root=root)
    return goal["id"], task["id"]


class TestMcpArchiveTools:
    def test_archive_round_trip_hides_and_restores(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal_id, _task_id = _seed_done_subtree(root)

        result = _call(milknado_archive_node, node_id=goal_id, project_root=root)
        assert result == {"archived": 2}

        tree = _call(milknado_todo_tree, project_root=root)
        assert tree == []
        full = _call(milknado_todo_tree, project_root=root, include_archived=True)
        assert [n["id"] for n in full] == [goal_id]
        assert _call(milknado_todo_tree, root_id=goal_id, project_root=root) == []
        explicit = _call(
            milknado_todo_tree,
            root_id=goal_id,
            project_root=root,
            include_archived=True,
        )
        assert [node["id"] for node in explicit] == [goal_id]

        restored = _call(milknado_unarchive_node, node_id=goal_id, project_root=root)
        assert restored == {"unarchived": 2}
        assert [n["id"] for n in _call(milknado_todo_tree, project_root=root)] == [goal_id]

    def test_archive_refuses_live_subtree(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="g", kind="goal", project_root=root)
        _call(
            milknado_todo_add,
            description="t",
            kind="task",
            parent_id=goal["id"],
            project_root=root,
        )
        with pytest.raises(ValueError, match="not eligible"):
            _call(milknado_archive_node, node_id=goal["id"], project_root=root)

    def test_archive_missing_node_fails(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="node 999 not found"):
            _call(milknado_archive_node, node_id=999, project_root=str(tmp_path))

    def test_unarchive_refuses_under_archived_ancestor(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal_id, task_id = _seed_done_subtree(root)
        _call(milknado_archive_node, node_id=goal_id, project_root=root)
        with pytest.raises(ValueError, match="unarchive the ancestor first"):
            _call(milknado_unarchive_node, node_id=task_id, project_root=root)


class TestMcpIncludeArchived:
    def test_graph_summary_filters_archived_by_default(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal_id, _ = _seed_done_subtree(root)
        before = _call(milknado_graph_summary, project_root=root)
        assert len(before["nodes"]) == 2

        _call(milknado_archive_node, node_id=goal_id, project_root=root)

        hidden = _call(milknado_graph_summary, project_root=root)
        assert hidden["nodes"] == []
        full = _call(milknado_graph_summary, project_root=root, include_archived=True)
        assert len(full["nodes"]) == 2

    def test_set_subtree_status_skips_archived(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal = _call(milknado_todo_add, description="g", kind="goal", project_root=root)
        live = _call(
            milknado_todo_add, description="live", parent_id=goal["id"], project_root=root
        )
        shelved = _call(
            milknado_todo_add, description="shelved", parent_id=goal["id"], project_root=root
        )
        _call(milknado_todo_set_status, node_id=shelved["id"], status="done", project_root=root)
        _call(milknado_archive_node, node_id=shelved["id"], project_root=root)

        result = _call(
            milknado_set_subtree_status, root_id=goal["id"], status="blocked", project_root=root
        )
        assert result == {"updated": 2}

        full = _call(milknado_todo_tree, project_root=root, include_archived=True)
        by_id = {n["id"]: n for n in full[0]["children"]}
        assert by_id[shelved["id"]]["status"] == "done"
        assert by_id[live["id"]]["status"] == "blocked"


class TestMcpSetStatusGuard:
    """Cross-layer: the archived-node status guard must bite through the MCP
    todo_set_status surface, not just the domain (spec: 'WHEN a status
    transition would change an archived node's status THE SYSTEM SHALL raise')."""

    def test_status_change_on_archived_node_raises_and_leaves_it_untouched(
        self, tmp_path: Path
    ) -> None:
        root = str(tmp_path)
        goal_id, task_id = _seed_done_subtree(root)
        _call(milknado_archive_node, node_id=goal_id, project_root=root)

        with pytest.raises(ValueError, match="is archived"):
            _call(milknado_todo_set_status, node_id=task_id, status="blocked", project_root=root)

        full = _call(milknado_todo_tree, project_root=root, include_archived=True)
        by_id = {n["id"]: n for n in full[0]["children"]}
        assert by_id[task_id]["status"] == "done"


class TestMcpRebalanceRestructure:
    """Cross-curd: restructure (curd 3) observable through the MCP rebalance
    surface — per-curd presses only exercised sweep/dry-run at this layer."""

    def test_orphans_grouped_under_find_or_create_inbox_and_inbox_reused(
        self, tmp_path: Path
    ) -> None:
        root = str(tmp_path)
        first = _call(milknado_todo_add, description="orphan one", project_root=root)
        second = _call(milknado_todo_add, description="orphan two", project_root=root)
        goal_id, _ = _seed_done_subtree(root)

        result = _call(milknado_rebalance, project_root=root, reap=False)
        # Sweep archived the DONE subtree before grouping; only live work moved.
        assert goal_id in result["archived_roots"]
        assert result["moved_tasks"] == 2
        inbox_id = result["inbox_id"]
        assert inbox_id is not None

        tree = _call(milknado_todo_tree, project_root=root)
        assert [n["id"] for n in tree] == [inbox_id]
        assert {c["id"] for c in tree[0]["children"]} == {first["id"], second["id"]}

        # A second run find-or-creates nothing: the Inbox is reused, no orphans remain.
        again = _call(milknado_rebalance, project_root=root, reap=False)
        assert again["inbox_id"] == inbox_id
        assert again["moved_tasks"] == 0

    def test_diamond_orphans_move_exactly_once_under_inbox(self, tmp_path: Path) -> None:
        """Two root tasks sharing one prerequisite (a diamond): every orphan is
        reparented exactly once under the Inbox, and the prereq edges onto the
        shared node survive the move (the tree render traverses prereq edges,
        so the shared node appears once as Inbox child and once per dependent)."""
        root = str(tmp_path)
        shared = _call(milknado_todo_add, description="shared prereq", project_root=root)
        left = _call(
            milknado_todo_add, description="left", prereqs=[shared["id"]], project_root=root
        )
        right = _call(
            milknado_todo_add, description="right", prereqs=[shared["id"]], project_root=root
        )

        result = _call(milknado_rebalance, project_root=root, reap=False)
        assert result["moved_tasks"] == 3
        inbox_id = result["inbox_id"]

        for task in (shared, left, right):
            assert (
                _call(milknado_get_node, node_id=task["id"], project_root=root)["parent_id"]
                == inbox_id
            )

        def _count(node: dict, target: int) -> int:
            return (node["id"] == target) + sum(_count(c, target) for c in node["children"])

        tree = _call(milknado_todo_tree, project_root=root)
        inbox = next(n for n in tree if n["id"] == inbox_id)
        # Parentage: each orphan is a direct Inbox child exactly once.
        assert sorted(c["id"] for c in inbox["children"]) == sorted(
            t["id"] for t in (shared, left, right)
        )
        # Prereq edges survived: both dependents still list the shared node.
        assert sum(_count(n, shared["id"]) for n in tree) == 3


class TestMcpRebalance:
    def test_dry_run_predicts_without_mutating(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal_id, _ = _seed_done_subtree(root)

        result = _call(milknado_rebalance, dry_run=True, project_root=root)
        assert "[dry-run] Would" in result["report"]
        # Nothing was swept for real: the subtree still shows in default reads.
        assert [n["id"] for n in _call(milknado_todo_tree, project_root=root)] == [goal_id]

    def test_real_sweep_archives_and_reports(self, tmp_path: Path) -> None:
        root = str(tmp_path)
        goal_id, _ = _seed_done_subtree(root)

        result = _call(milknado_rebalance, project_root=root, reap=False)
        assert goal_id in result["archived_roots"]
        assert _call(milknado_todo_tree, project_root=root) == []

    def test_dry_run_on_uninitialized_project_creates_nothing(self, tmp_path: Path) -> None:
        result = _call(milknado_rebalance, dry_run=True, project_root=str(tmp_path))
        assert list(result["archived_roots"]) == []
        assert not (tmp_path / ".milknado").exists()


def _init_and_seed_done_goal(project_dir: Path) -> tuple[int, int]:
    """init a project, seed goal+task both DONE via the facade; return (goal, task) ids."""
    runner.invoke(app, ["init", str(project_dir)])
    config = default_config(project_dir)
    graph = MikadoGraph(config.db_path)
    goal = graph.add_node("goal")
    task = graph.add_node("task", parent_id=goal.id)
    _complete(graph, task.id)
    _complete(graph, goal.id)
    graph.close()
    return goal.id, task.id


class TestCliGraphArchive:
    def test_archive_then_unarchive(self, tmp_path: Path) -> None:
        goal_id, _ = _init_and_seed_done_goal(tmp_path)
        result = runner.invoke(
            app, ["graph", "archive", str(goal_id), "--project-root", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Archived 2 node(s)" in result.output

        result = runner.invoke(
            app, ["graph", "unarchive", str(goal_id), "--project-root", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Unarchived 2 node(s)" in result.output

    def test_archive_refuses_live_subtree(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        config = default_config(tmp_path)
        graph = MikadoGraph(config.db_path)
        goal = graph.add_node("goal")
        task = graph.add_node("task", parent_id=goal.id)
        graph.close()

        result = runner.invoke(
            app, ["graph", "archive", str(goal.id), "--project-root", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "nodes not eligible for archive" in result.output
        assert str(goal.id) in result.output
        assert str(task.id) in result.output

    def test_archive_missing_node_exit_1(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        result = runner.invoke(app, ["graph", "archive", "999", "--project-root", str(tmp_path)])
        assert result.exit_code == 1
        assert "999" in result.output

    def test_unarchive_under_archived_ancestor_refused(self, tmp_path: Path) -> None:
        goal_id, task_id = _init_and_seed_done_goal(tmp_path)
        runner.invoke(app, ["graph", "archive", str(goal_id), "--project-root", str(tmp_path)])
        result = runner.invoke(
            app, ["graph", "unarchive", str(task_id), "--project-root", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "unarchive the ancestor first" in result.output


class TestCliStatusAll:
    def test_archived_hidden_by_default_marker_with_all(self, tmp_path: Path) -> None:
        goal_id, _ = _init_and_seed_done_goal(tmp_path)
        runner.invoke(app, ["graph", "archive", str(goal_id), "--project-root", str(tmp_path)])

        hidden = runner.invoke(app, ["status", str(tmp_path)])
        assert "goal" not in hidden.output

        shown = runner.invoke(app, ["status", str(tmp_path), "--all"])
        assert "goal" in shown.output
        assert "[archived]" in shown.output


class TestCliRebalance:
    def test_dry_run_prints_plan_and_mutates_nothing(self, tmp_path: Path) -> None:
        goal_id, _ = _init_and_seed_done_goal(tmp_path)
        result = runner.invoke(app, ["rebalance", str(tmp_path), "--dry-run"])
        assert result.exit_code == 0
        assert "[dry-run] Would" in result.output

        config = default_config(tmp_path)
        graph = MikadoGraph(config.db_path)
        assert graph.get_node(goal_id).archived_at is None  # type: ignore[union-attr]
        graph.close()

    def test_dry_run_plan_matches_apply(self, tmp_path: Path) -> None:
        """CLI dry-run/apply parity: the plan names exactly the roots the real
        run archives, and a post-apply dry-run has nothing left to sweep."""
        goal_id, _ = _init_and_seed_done_goal(tmp_path)

        dry = runner.invoke(app, ["rebalance", str(tmp_path), "--dry-run"])
        assert dry.exit_code == 0
        assert f"roots: {goal_id}" in dry.output

        applied = runner.invoke(app, ["rebalance", str(tmp_path), "--no-reap"])
        assert applied.exit_code == 0
        assert "[dry-run]" not in applied.output
        assert f"roots: {goal_id}" in applied.output

        after = runner.invoke(app, ["rebalance", str(tmp_path), "--dry-run"])
        assert after.exit_code == 0
        assert "roots: none" in after.output

    def test_real_run_sweeps(self, tmp_path: Path) -> None:
        goal_id, _ = _init_and_seed_done_goal(tmp_path)
        result = runner.invoke(app, ["rebalance", str(tmp_path), "--no-reap"])
        assert result.exit_code == 0

        config = default_config(tmp_path)
        graph = MikadoGraph(config.db_path)
        assert graph.get_node(goal_id).archived_at is not None  # type: ignore[union-attr]
        graph.close()


class TestCliSweepThenUnarchive:
    """Cross-curd interaction: a subtree shelved by `rebalance` (curd 3) must be
    restorable through `graph unarchive` (curd 1 wiring), and `status --all`
    must render the swept tree with the archived marker in between."""

    def test_swept_subtree_marked_by_status_all_and_restored_by_unarchive(
        self, tmp_path: Path
    ) -> None:
        goal_id, _ = _init_and_seed_done_goal(tmp_path)

        swept = runner.invoke(app, ["rebalance", str(tmp_path), "--no-reap"])
        assert swept.exit_code == 0

        hidden = runner.invoke(app, ["status", str(tmp_path)])
        assert "goal" not in hidden.output
        shown = runner.invoke(app, ["status", str(tmp_path), "--all"])
        assert "goal" in shown.output
        assert "[archived]" in shown.output

        restored = runner.invoke(
            app, ["graph", "unarchive", str(goal_id), "--project-root", str(tmp_path)]
        )
        assert restored.exit_code == 0

        visible = runner.invoke(app, ["status", str(tmp_path)])
        assert "goal" in visible.output
        assert "[archived]" not in visible.output


class TestCliStatusEmptyGraphHint:
    """Regression (post-merge F4): when every node is archived, the default
    `status` view must say so instead of implying the project is unstarted."""

    def test_all_archived_default_status_prints_archived_hint(self, tmp_path: Path) -> None:
        goal_id, _ = _init_and_seed_done_goal(tmp_path)
        result = runner.invoke(
            app, ["graph", "archive", str(goal_id), "--project-root", str(tmp_path)]
        )
        assert result.exit_code == 0

        shown = runner.invoke(app, ["status", str(tmp_path)])

        assert shown.exit_code == 0
        assert "No nodes in graph" in shown.output
        assert "archived node(s) — use --all" in shown.output

    def test_truly_empty_graph_keeps_original_status_message(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])

        shown = runner.invoke(app, ["status", str(tmp_path)])

        assert shown.exit_code == 0
        assert "No nodes in graph. Run milknado plan to start." in shown.output
        assert "archived" not in shown.output
