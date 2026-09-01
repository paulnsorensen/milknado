from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from milknado.domains.common import MikadoNode, NodeKind, NodeSpec, NodeStatus
from milknado.domains.common.errors import InvalidTransition
from milknado.domains.graph import MikadoGraph
from milknado.domains.graph.status_flow import (
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


def test_bulk_preflight_leaves_every_node_unchanged(graph: MikadoGraph) -> None:
    root = graph.add_node("root")
    done = graph.add_node("done", parent_id=root.id)
    graph.mark_running(done.id)
    graph.mark_done(done.id)

    with pytest.raises(InvalidTransition):
        _ = graph.set_subtree_status(root.id, NodeStatus.PENDING)

    root_node = graph.get_node(root.id)
    done_node = graph.get_node(done.id)
    assert root_node is not None
    assert done_node is not None
    assert root_node.status is NodeStatus.PENDING
    assert done_node.status is NodeStatus.DONE


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
    inner_node = graph.get_node(inner.id)
    outer_node = graph.get_node(outer.id)
    assert inner_node is not None
    assert outer_node is not None
    assert inner_node.status is NodeStatus.DONE
    assert outer_node.status is NodeStatus.DONE


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

    goal_node = graph.get_node(goal.id)
    assert goal_node is not None
    assert goal_node.status is NodeStatus.DONE
