"""Stale-worker recovery of a native group attempt resolves its reservation once."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from milknado.domains.common import NodeStatus
from milknado.domains.coordinator.model import CoordinatorSession
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.dispatch.reconcile import fail_stale_running_runs
from milknado.domains.graph import GroupWorkspace, MikadoGraph, TaskAttempt
from milknado.loop._process_adapter import ProcessAdapter

pytestmark = pytest.mark.real_process

_Vanished = tuple[MikadoGraph, CoordinatorWorkflow, CoordinatorSession, str, int, TaskAttempt]


def _vanished_native_run(root: Path) -> _Vanished:
    graph = MikadoGraph(root / "graph.db")
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=2)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    group = workflow.create_group(
        session, "main", (task.id,), GroupWorkspace(str(root / "group"), "branch", None)
    )
    first = workflow.dispatch_task(session, group.id, task.id, "run-1")
    _ = workflow.acknowledge_launch(session, first)
    attempt = first.attempt
    assert graph.task_attempts.confirm_execution(
        attempt.attempt_id, attempt.attempt_id, attempt.attempt_id
    )
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    assert process.wait(timeout=5) == 0
    graph.runs.set_pid(attempt.attempt_id, process.pid)
    return graph, workflow, session, group.id, task.id, attempt


def test_recovery_counts_one_failure_and_frees_the_reservation(tmp_path: Path) -> None:
    graph, workflow, session, group_id, node_id, attempt = _vanished_native_run(tmp_path)

    flipped = fail_stale_running_runs(graph, node_id, ProcessAdapter())

    assert len(flipped) == 1
    assert graph.task_attempts.failed_count(node_id) == 1
    assert graph.groups.active_attempt(group_id) is None
    assert graph.groups.task_result(node_id) is None
    assert fail_stale_running_runs(graph, node_id, ProcessAdapter()) == []
    assert graph.task_attempts.failed_count(node_id) == 1
    retry = workflow.dispatch_task(session, group_id, node_id, "run-2")
    assert retry.attempt.attempt_id != attempt.attempt_id
    graph.close()


def test_recovery_leaves_a_newer_reservation_and_counts_nothing(tmp_path: Path) -> None:
    graph, _, _, group_id, node_id, attempt = _vanished_native_run(tmp_path)
    with sqlite3.connect(graph.db_path) as conn:
        _ = conn.execute(
            "UPDATE execution_groups SET active_attempt_id = 'newer' WHERE id = ?", (group_id,)
        )

    assert fail_stale_running_runs(graph, node_id, ProcessAdapter()) == []

    assert graph.task_attempts.failed_count(node_id) == 0
    newer = graph.groups.active_attempt(group_id)
    assert newer is not None and newer.attempt_id == "newer"
    current = graph.get_node(node_id)
    assert current is not None and current.status is NodeStatus.RUNNING
    record = graph.task_attempts.get(attempt.attempt_id)
    assert record is not None and record.outcome == "active"
    graph.close()
