"""Native coordinator failures with real worker invocations and launch-failure evidence."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from milknado.domains.common import MikadoNode, NodeStatus, RunResult
from milknado.domains.coordinator.model import CoordinatorSession
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import ExecutionGroup, GroupWorkspace, MikadoGraph


class _Fixture(NamedTuple):
    graph: MikadoGraph
    workflow: CoordinatorWorkflow
    session: CoordinatorSession
    task: MikadoNode
    group: ExecutionGroup


def _setup(tmp_path: Path, budget: int) -> _Fixture:
    graph = MikadoGraph(tmp_path / "graph.db")
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=budget)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    workspace = GroupWorkspace(str(tmp_path / "group"), "branch", "provider")
    group = workflow.create_group(session, "main", (task.id,), workspace)
    return _Fixture(graph, workflow, session, task, group)


def test_failure_after_worker_invocation_bind_requeues_then_blocks(tmp_path: Path) -> None:
    fx = _setup(tmp_path, 2)
    for number, expected in ((1, "pending"), (2, "blocked")):
        handoff = fx.workflow.dispatch_task(fx.session, fx.group.id, fx.task.id, f"run-{number}")
        _ = fx.workflow.acknowledge_launch(fx.session, handoff)
        attempt_id = handoff.attempt.attempt_id
        identity = (attempt_id, attempt_id, attempt_id, f"worker-invocation-{number}")
        assert fx.graph.task_attempts.bind_worker_invocation(identity)
        outcome = NodeLoopOutcome(fx.task.id, False, "boom")
        fx.workflow.finish_task(fx.session, handoff.attempt, outcome)
        node = fx.graph.get_node(fx.task.id)
        assert node is not None and node.status.value == expected
        assert fx.graph.task_attempts.failed_count(fx.task.id) == number
    fx.graph.close()


def test_reopened_exhausted_group_member_is_dispatchable_again(tmp_path: Path) -> None:
    fx = _setup(tmp_path, 1)
    handoff = fx.workflow.dispatch_task(fx.session, fx.group.id, fx.task.id, "run-1")
    _ = fx.workflow.acknowledge_launch(fx.session, handoff)
    attempt_id = handoff.attempt.attempt_id
    assert fx.graph.task_attempts.bind_worker_invocation(
        (attempt_id, attempt_id, attempt_id, "invocation")
    )
    fx.workflow.finish_task(
        fx.session, handoff.attempt, NodeLoopOutcome(fx.task.id, False, "boom")
    )
    node = fx.graph.get_node(fx.task.id)
    assert node is not None and node.status is NodeStatus.FAILED

    _ = fx.graph.set_todo_status(fx.task.id, NodeStatus.PENDING)

    again = fx.workflow.dispatch_task(fx.session, fx.group.id, fx.task.id, "run-2")
    assert fx.workflow.dispatch_state(again.attempt.attempt_id) == "awaiting_launch"
    fx.graph.close()


def test_rejected_launch_failure_after_partial_acknowledge_leaves_no_evidence(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path, 2)
    handoff = fx.workflow.dispatch_task(fx.session, fx.group.id, fx.task.id, "run-1")
    attempt_id = handoff.attempt.attempt_id
    fx.graph.runs.start(attempt_id, fx.task.id, "wt", "2026-01-01T00:00:00+00:00", None)
    fx.graph.runs.finish(attempt_id, RunResult("failed", 1, False, "2026-01-01T00:00:01+00:00"))
    with pytest.raises(ValueError, match="group run identity conflicts"):
        _ = fx.workflow.acknowledge_launch(fx.session, handoff)
    assert fx.workflow.dispatch_state(attempt_id) == "awaiting_launch"

    with pytest.raises(ValueError, match="changed owner"):
        fx.workflow.fail_launch(fx.session, handoff, "spawn failed")

    assert fx.workflow.dispatch_state(attempt_id) == "awaiting_launch"
    record = fx.graph.task_attempts.get(attempt_id)
    assert record is not None and record.start_state != "startup_failed"
    node = fx.graph.get_node(fx.task.id)
    assert node is not None and node.status is NodeStatus.RUNNING
    fx.graph.close()
