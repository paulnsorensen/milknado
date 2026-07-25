from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from milknado.domains.common import MikadoNode, NodeKind, NodeSpec, NodeStatus
from milknado.domains.common.errors import InvalidTransition
from milknado.domains.graph import MikadoGraph
from milknado.domains.graph.status_flow import (
    latest_verify_ok,
    subtree_post_order,
    validate_todo_status,
)

_ALL = {
    NodeStatus.PENDING,
    NodeStatus.RUNNING,
    NodeStatus.BLOCKED,
    NodeStatus.FAILED,
    NodeStatus.DONE,
}
_ALLOWED = {
    NodeStatus.PENDING: _ALL,
    NodeStatus.RUNNING: _ALL,
    NodeStatus.BLOCKED: _ALL - {NodeStatus.FAILED},
    NodeStatus.FAILED: {NodeStatus.PENDING, NodeStatus.FAILED},
    NodeStatus.DONE: {NodeStatus.DONE},
}


@pytest.mark.parametrize(
    ("current", "target", "allowed"),
    [
        (current, target, target in allowed)
        for current, allowed in _ALLOWED.items()
        for target in NodeStatus
    ],
)
def test_todo_transition_matrix(current: NodeStatus, target: NodeStatus, allowed: bool) -> None:
    node = MikadoNode(id=1, description="node", status=current)
    if allowed:
        validate_todo_status(node, target)
    else:
        with pytest.raises(InvalidTransition):
            validate_todo_status(node, target)


@pytest.mark.parametrize(
    "deposit",
    [None, "not-json", "[]", "true", "false", "1", '{"ok": "true"}'],
)
def test_verification_deposits_fail_closed(deposit: str | None) -> None:
    class Graph:
        def latest_run_message(self, _run_id: str, _role: str) -> str | None:
            return deposit

    assert latest_verify_ok(Graph(), "run") is False


def test_bulk_preflight_leaves_every_node_unchanged(graph: MikadoGraph) -> None:
    root = graph.add_node("root")
    done = graph.add_node("done", parent_id=root.id)
    graph.mark_running(done.id)
    graph.mark_done(done.id)

    with pytest.raises(InvalidTransition):
        graph.set_subtree_status(root.id, NodeStatus.PENDING)

    assert graph.get_node(root.id).status is NodeStatus.PENDING
    assert graph.get_node(done.id).status is NodeStatus.DONE


def test_post_order_deduplicates_cycles_and_deep_graph(graph: MikadoGraph) -> None:
    root = graph.add_node("root")
    current = root
    for _ in range(1_100):
        current = graph.add_node("child", parent_id=current.id)

    nodes = subtree_post_order(graph.get_children_map(), root)
    assert len(nodes) == 1_101
    assert nodes[-1].id == root.id


def test_reconcile_completed_goals_recurses_upward(graph: MikadoGraph) -> None:
    outer = graph.add_node("outer", spec=NodeSpec(kind=NodeKind.GOAL))
    inner = graph.add_node("inner", parent_id=outer.id, spec=NodeSpec(kind=NodeKind.GOAL))
    task = graph.add_node("task", parent_id=inner.id)
    graph.mark_running(task.id)
    graph.mark_done(task.id)

    assert graph.reconcile_completed_goals() == 2
    assert graph.get_node(inner.id).status is NodeStatus.DONE
    assert graph.get_node(outer.id).status is NodeStatus.DONE


def test_run_startup_reconciles_completed_goal_through_cli(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    from milknado.cli.run import run

    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    task = graph.add_node("task", parent_id=goal.id)
    graph.mark_running(task.id)
    graph.mark_done(task.id)

    with (
        patch("milknado.cli.run._load_or_default", return_value=(object(), [])),
        patch("milknado.cli.run._ensure_db", return_value=graph),
        patch("milknado.app.run.resolve_feature_branch", return_value="feature"),
        patch("milknado.app.run.check_protected_branch", return_value=None),
        patch("milknado.cli.run._is_interactive_terminal", return_value=False),
        patch("milknado.domains.dispatch.reconcile_orphaned_runs"),
        patch("milknado.domains.execution.get_dispatchable_nodes", return_value=[]),
    ):
        run(project_root=tmp_path)

    assert graph.get_node(goal.id).status is NodeStatus.DONE
