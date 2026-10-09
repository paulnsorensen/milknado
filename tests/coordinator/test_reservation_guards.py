from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.common import NodeStatus, WorkerIdentity, WorkerOwner
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import GroupWorkspace, MikadoGraph


def test_reserved_launch_rechecks_prerequisites(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        prerequisite = graph.add_node("Prerequisite", session.goal_id)
        task = graph.add_node("Task", session.goal_id)
        _ = graph.add_edge(task.id, prerequisite.id)
        assert graph.claim_node(
            prerequisite.id, "prerequisite-run", now="2026-01-01T00:00:00+00:00"
        )
        graph.mark_done(prerequisite.id)
        group = workflow.create_group(
            session, "main", (task.id,), GroupWorkspace("/tmp/task", "task", "provider")
        )
        handoff = workflow.dispatch_task(session, group.id, task.id, "run")
        with conn:
            _ = conn.execute(
                "UPDATE nodes SET status = 'pending' WHERE id = ?", (prerequisite.id,)
            )

        with pytest.raises(ValueError, match="incomplete external prerequisite"):
            _ = workflow.acknowledge_launch(session, handoff)
        assert graph.groups.active_attempt(group.id) == handoff.attempt
        assert graph.groups.task_result(task.id) is None
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.PENDING
        assert workflow.dispatch_state(handoff.attempt.attempt_id) == "awaiting_launch"

        with conn:
            _ = conn.execute("UPDATE nodes SET status = 'done' WHERE id = ?", (prerequisite.id,))
        launched = workflow.acknowledge_launch(session, handoff)
        assert launched.state == "launched"
        assert graph.groups.active_attempt(group.id) == handoff.attempt
        assert graph.groups.task_result(task.id) is None
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.RUNNING
    graph.close()


def test_running_launch_retry_ignores_prerequisite_drift(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        prerequisite = graph.add_node("Prerequisite", session.goal_id)
        task = graph.add_node("Task", session.goal_id)
        _ = graph.add_edge(task.id, prerequisite.id)
        assert graph.claim_node(
            prerequisite.id, "prerequisite-run", now="2026-01-01T00:00:00+00:00"
        )
        graph.mark_done(prerequisite.id)
        group = workflow.create_group(
            session, "main", (task.id,), GroupWorkspace("/tmp/task", "task", "provider")
        )
        handoff = workflow.dispatch_task(session, group.id, task.id, "run")
        launched = workflow.acknowledge_launch(session, handoff)
        with conn:
            _ = conn.execute(
                "UPDATE nodes SET status = 'pending' WHERE id = ?", (prerequisite.id,)
            )
        retried = workflow.acknowledge_launch(session, handoff)
        assert retried == launched
        assert graph.groups.active_attempt(group.id) == handoff.attempt
        assert graph.groups.task_result(task.id) is None
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.RUNNING
        assert workflow.dispatch_state(handoff.attempt.attempt_id) == "launched"
    graph.close()


def test_launch_failure_waits_for_node_worker_cleanup(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        group = workflow.create_group(
            session, "main", (task.id,), GroupWorkspace("/tmp/task", "task", "provider")
        )
        handoff = workflow.dispatch_task(session, group.id, task.id, "run")
        graph.runs.start("worker-run", task.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        graph.runs.record_worker(
            WorkerOwner("worker-run", 999999, 123.5, "worker-run", task.id),
            WorkerIdentity("inv-1", 999998, 999998, 123.5),
        )

        with pytest.raises(ValueError, match="unresolved worker"):
            workflow.fail_launch(session, handoff, "spawn failed")
        assert graph.groups.active_attempt(group.id) == handoff.attempt
        assert graph.groups.task_result(task.id) is None
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.PENDING
        assert workflow.dispatch_state(handoff.attempt.attempt_id) == "awaiting_launch"

        graph.runs.end_worker("inv-1", 0)
        workflow.fail_launch(session, handoff, "spawn failed")
        assert graph.groups.active_attempt(group.id) is None
        assert graph.groups.task_result(task.id) == ("failed", "spawn failed")
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.PENDING
        assert workflow.dispatch_state(handoff.attempt.attempt_id) == "launch_failed"
    graph.close()
