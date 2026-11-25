"""Public native replay fences after ownership and outcome changes."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.coordinator.model import CoordinatorSession
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import GroupWorkspace, MikadoGraph, TaskAttempt, TaskOutcome


def _launched(
    path: Path,
) -> tuple[MikadoGraph, CoordinatorWorkflow, CoordinatorSession, TaskAttempt]:
    graph = MikadoGraph(path)
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=3)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    group = workflow.create_group(
        session,
        "main",
        (task.id,),
        GroupWorkspace(str(path.parent / "group"), "branch", "provider"),
    )
    handoff = workflow.dispatch_task(session, group.id, task.id, "run")
    _ = workflow.acknowledge_launch(session, handoff)
    return graph, workflow, session, handoff.attempt


def _assert_no_replay_write(
    graph: MikadoGraph, workflow: CoordinatorWorkflow, attempt: TaskAttempt
) -> None:
    run = graph.runs.get(attempt.attempt_id)
    assert run is not None and run["status"] == "running"
    assert workflow.dispatch_state(attempt.attempt_id) == "launched"
    assert graph.task_attempts.failed_count(attempt.node_id) == 0


def test_replay_refuses_changed_ledger_owner(tmp_path: Path) -> None:
    graph, workflow, session, attempt = _launched(tmp_path / "graph.db")
    assert graph.replace_run_id(attempt.node_id, attempt.attempt_id, "new-owner")
    assert graph.task_attempts.bind_invocation(attempt.attempt_id, "new-owner", "new-owner")
    with pytest.raises(ValueError, match="native task attempt owner fence lost"):
        workflow.finish_task(session, attempt, NodeLoopOutcome(attempt.node_id, False, "stale"))
    record = graph.task_attempts.get(attempt.attempt_id)
    assert record is not None and record.owner_run_id == "new-owner"
    assert record.outcome == "active"
    assert graph.groups.task_result(attempt.node_id) is None
    _assert_no_replay_write(graph, workflow, attempt)
    graph.close()


def test_failure_refuses_changed_node_owner(tmp_path: Path) -> None:
    graph, workflow, session, attempt = _launched(tmp_path / "graph.db")
    assert graph.replace_run_id(attempt.node_id, attempt.attempt_id, "new-owner")
    with pytest.raises(RuntimeError, match="task attempt fence lost for run"):
        workflow.finish_task(session, attempt, NodeLoopOutcome(attempt.node_id, False, "stale"))
    node = graph.get_node(attempt.node_id)
    assert node is not None and node.run_id == "new-owner"
    record = graph.task_attempts.get(attempt.attempt_id)
    assert record is not None and record.outcome == "active"
    assert graph.groups.task_result(attempt.node_id) is None
    _assert_no_replay_write(graph, workflow, attempt)
    graph.close()


def test_nonfailure_replay_refuses_conflicting_settled_ledger(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph, workflow, session, attempt = _launched(path)
    graph.groups.finish_task(attempt, TaskOutcome("done", "result"))
    assert graph.task_attempts.record_nonfailure(
        attempt.attempt_id, attempt.attempt_id, "preserved"
    )
    graph.close()
    graph = MikadoGraph(path)
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=3)
    with pytest.raises(ValueError, match="worker result conflicts with recorded task attempt"):
        workflow.finish_task(session, attempt, NodeLoopOutcome(attempt.node_id, True, "result"))
    record = graph.task_attempts.get(attempt.attempt_id)
    assert record is not None and record.outcome == "preserved"
    assert graph.groups.task_result(attempt.node_id) == ("done", "result")
    _assert_no_replay_write(graph, workflow, attempt)
    graph.close()
