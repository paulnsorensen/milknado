"""Native attempt retry, replay, hold, and transaction behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import MikadoNode
from milknado.domains.coordinator.model import CoordinatorSession
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import (
    GoalReviewRequest,
    GroupWorkspace,
    MikadoGraph,
    TaskAttempt,
)


def _node_status(graph: MikadoGraph, node_id: int) -> str:
    node = graph.get_node(node_id)
    assert node is not None
    return node.status.value


def _run_status(graph: MikadoGraph, run_id: str) -> str:
    run = graph.runs.get(run_id)
    assert run is not None
    return run["status"]


def _attempt_outcome(graph: MikadoGraph, attempt_id: str) -> str:
    attempt = graph.task_attempts.get(attempt_id)
    assert attempt is not None
    return attempt.outcome


def _node_row(graph: MikadoGraph, node_id: int) -> tuple[str, str]:
    row = cast(
        sqlite3.Row | None,
        graph.group_connection.execute(
            "SELECT status, run_id FROM nodes WHERE id = ?", (node_id,)
        ).fetchone(),
    )
    assert row is not None
    return cast(str, row[0]), cast(str, row[1])


def _reservation_id(graph: MikadoGraph, group_id: str) -> str:
    row = cast(
        sqlite3.Row | None,
        graph.group_connection.execute(
            "SELECT active_attempt_id FROM execution_groups WHERE id = ?", (group_id,)
        ).fetchone(),
    )
    assert row is not None
    return cast(str, row[0])


def _launched_task(
    graph: MikadoGraph, root: Path, run_id: str
) -> tuple[CoordinatorWorkflow, CoordinatorSession, MikadoNode, TaskAttempt]:
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=3)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    group = workflow.create_group(
        session,
        "main",
        (task.id,),
        GroupWorkspace(str(root / "group"), "branch", "provider"),
    )
    handoff = workflow.dispatch_task(session, group.id, task.id, run_id)
    _ = workflow.acknowledge_launch(session, handoff)
    return workflow, session, task, handoff.attempt


def test_native_exhaustion_replays_only_final_result(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=3)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    group = workflow.create_group(
        session,
        "main",
        (task.id,),
        GroupWorkspace(str(tmp_path / "group"), "branch", "provider"),
    )
    for number in range(1, 4):
        handoff = workflow.dispatch_task(session, group.id, task.id, f"run-{number}")
        _ = workflow.acknowledge_launch(session, handoff)
        outcome = NodeLoopOutcome(task.id, False, f"error-{number}")
        workflow.finish_task(session, handoff.attempt, outcome)
        assert graph.task_attempts.failed_count(task.id) == number
        expected = "pending" if number < 3 else "blocked"
        assert _node_status(graph, task.id) == expected
        assert graph.groups.task_result(task.id) == (
            None if number < 3 else ("blocked", "error-3")
        )
        workflow.finish_task(session, handoff.attempt, outcome)
        assert graph.task_attempts.failed_count(task.id) == number
    graph.close()


def test_native_failure_replay_finishes_dispatch_without_touching_new_reservation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    workflow, session, task, first = _launched_task(graph, tmp_path, "run-1")
    group_id = first.group_id
    attempt_id = first.attempt_id
    _ = graph.group_connection.execute(
        """CREATE TEMP TRIGGER reject_dispatch_finish
        BEFORE UPDATE OF state ON coordinator_dispatches
        WHEN NEW.state = 'finished'
        BEGIN SELECT RAISE(ABORT, 'injected coordinator failure'); END"""
    )
    outcome = NodeLoopOutcome(task.id, False, "failed once")
    with pytest.raises(sqlite3.IntegrityError, match="injected coordinator failure"):
        workflow.finish_task(session, first, outcome)
    graph.close()

    graph = MikadoGraph(path)
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=3)
    assert workflow.dispatch_state(attempt_id) == "launched"
    assert graph.task_attempts.failed_count(task.id) == 1
    second = workflow.dispatch_task(session, group_id, task.id, "run-2")
    assert _reservation_id(graph, group_id) == second.attempt.attempt_id
    workflow.finish_task(session, first, outcome)
    assert workflow.dispatch_state(attempt_id) == "finished"
    assert graph.task_attempts.failed_count(task.id) == 1
    assert _reservation_id(graph, group_id) == second.attempt.attempt_id
    with pytest.raises(ValueError, match="worker result conflicts"):
        workflow.finish_task(session, first, NodeLoopOutcome(task.id, False, "conflict"))
    assert _reservation_id(graph, group_id) == second.attempt.attempt_id
    graph.close()


@pytest.mark.parametrize("preserved", [False, True])
def test_native_nonfailure_replay_recovers_canonical_outcome(
    tmp_path: Path, preserved: bool
) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    workflow, session, task, attempt = _launched_task(graph, tmp_path, "run")
    attempt_id = attempt.attempt_id
    _ = graph.group_connection.execute(
        """CREATE TEMP TRIGGER reject_nonfailure_ledger_update
        BEFORE UPDATE OF outcome ON task_attempts
        WHEN NEW.outcome IN ('done', 'preserved')
        BEGIN SELECT RAISE(ABORT, 'injected ledger failure'); END"""
    )
    outcome = NodeLoopOutcome(task.id, not preserved, "result", ownership_preserved=preserved)
    with pytest.raises(sqlite3.IntegrityError, match="injected ledger failure"):
        workflow.finish_task(session, attempt, outcome)
    graph.group_connection.rollback()
    graph.close()

    graph = MikadoGraph(path)
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=3)
    expected_status = "blocked" if preserved else "done"
    assert workflow.dispatch_state(attempt_id) == "launched"
    assert graph.groups.task_result(task.id) == (expected_status, "result")
    assert _attempt_outcome(graph, attempt_id) == "active"
    with pytest.raises(ValueError, match="worker result conflicts"):
        workflow.finish_task(
            session,
            attempt,
            NodeLoopOutcome(task.id, not preserved, "conflict", ownership_preserved=preserved),
        )
    assert _attempt_outcome(graph, attempt_id) == "active"
    workflow.finish_task(session, attempt, outcome)
    assert workflow.dispatch_state(attempt_id) == "finished"
    assert _attempt_outcome(graph, attempt_id) == ("preserved" if preserved else "done")
    assert _run_status(graph, attempt_id) == expected_status
    assert graph.task_attempts.failed_count(task.id) == 0
    workflow.finish_task(session, attempt, outcome)
    assert graph.task_attempts.failed_count(task.id) == 0
    graph.close()


def test_native_worker_failure_counts_without_manual_start_confirmation(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=2)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    group = workflow.create_group(
        session,
        "main",
        (task.id,),
        GroupWorkspace(str(tmp_path / "group"), "branch", "provider"),
    )
    for number, expected in ((1, "pending"), (2, "blocked")):
        handoff = workflow.dispatch_task(session, group.id, task.id, f"run-{number}")
        _ = workflow.acknowledge_launch(session, handoff)
        outcome = NodeLoopOutcome(task.id, False, "failed")
        workflow.finish_task(session, handoff.attempt, outcome)
        assert _node_status(graph, task.id) == expected
        assert graph.task_attempts.failed_count(task.id) == number
    graph.close()


def test_native_failure_transaction_rolls_back_every_row(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=3)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    group = workflow.create_group(
        session,
        "main",
        (task.id,),
        GroupWorkspace(str(tmp_path / "group"), "branch", "provider"),
    )
    handoff = workflow.dispatch_task(session, group.id, task.id, "run")
    _ = workflow.acknowledge_launch(session, handoff)
    attempt_id = handoff.attempt.attempt_id
    before = _node_row(graph, task.id)

    _ = graph.group_connection.execute(
        """CREATE TEMP TRIGGER reject_final_ledger_update
        BEFORE UPDATE OF outcome ON task_attempts
        WHEN NEW.outcome = 'failed' AND EXISTS (
            SELECT 1 FROM execution_groups AS g
            JOIN execution_group_tasks AS t ON t.group_id = g.id
            WHERE t.node_id = NEW.node_id AND g.active_attempt_id IS NULL
        )
        BEGIN SELECT RAISE(ABORT, 'injected late ledger failure'); END"""
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected late ledger failure"):
        workflow.finish_task(session, handoff.attempt, NodeLoopOutcome(task.id, False, "fail"))
    graph.close()

    graph = MikadoGraph(path)
    assert _node_row(graph, task.id) == before
    assert _run_status(graph, attempt_id) == "running"
    assert _attempt_outcome(graph, attempt_id) == "active"
    assert graph.task_attempts.failed_count(task.id) == 0
    assert graph.groups.task_result(task.id) is None
    assert _reservation_id(graph, group.id) == attempt_id
    graph.close()


def test_native_retry_still_requires_goal_review_admission(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=2)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    group = workflow.create_group(
        session,
        "main",
        (task.id,),
        GroupWorkspace(str(tmp_path / "group"), "branch", "provider"),
    )
    handoff = workflow.dispatch_task(session, group.id, task.id, "run-1")
    _ = workflow.acknowledge_launch(session, handoff)
    workflow.finish_task(session, handoff.attempt, NodeLoopOutcome(task.id, False, "failed"))
    assert _node_status(graph, task.id) == "pending"
    _ = graph.request_goal_review(
        GoalReviewRequest(
            goal_id=session.goal_id,
            goal_revision="sha256:goal-contract",
            evidence="New evidence changes the outcome.",
            proposed_change="Change the agreed goal.",
            affected_node_ids=(task.id,),
            reviewer="worker",
        )
    )
    with pytest.raises(ValueError, match="goal review pauses task dispatch"):
        _ = workflow.dispatch_task(session, group.id, task.id, "run-2")
    assert graph.groups.active_attempt(group.id) is None
    assert graph.task_attempts.failed_count(task.id) == 1
    graph.close()
