"""Native coordinator failures with real worker invocations and launch-failure evidence."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from milknado.domains.common import MikadoNode
from milknado.domains.coordinator._task_attempt_workflow import link_task_run
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


def test_fail_launch_records_startup_failed_when_attempt_row_exists(tmp_path: Path) -> None:
    fx = _setup(tmp_path, 2)
    handoff = fx.workflow.dispatch_task(fx.session, fx.group.id, fx.task.id, "run-1")
    fx.graph.groups.launch_reserved_task(handoff.attempt)
    link_task_run(fx.graph, handoff.attempt, 2)
    with pytest.raises(ValueError, match="changed owner"):
        fx.workflow.fail_launch(fx.session, handoff, "spawn failed")
    record = fx.graph.task_attempts.get(handoff.attempt.attempt_id)
    assert record is not None
    assert record.start_state == "startup_failed"
    fx.graph.close()
