from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import NodeStatus
from milknado.domains.coordinator import EntityLink
from milknado.domains.coordinator.journal import control_history
from milknado.domains.coordinator.persistence import link_entity, links_for_session
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import GoalReviewRequest, GroupWorkspace, MikadoGraph


def test_finish_rejects_foreign_goal_and_launch_failure_is_visible(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        owner = workflow.start_goal("Owner", "codex")
        foreign = workflow.start_goal("Foreign", "codex")
        task = graph.add_node("Task", owner.goal_id)
        group = workflow.create_group(
            owner,
            "owner-graph",
            (task.id,),
            GroupWorkspace("/tmp/owner", "owner", "provider-owner"),
        )
        handoff = workflow.dispatch_task(owner, group.id, task.id, "owner-run")
        assert handoff.state == "awaiting_launch"
        with pytest.raises(ValueError, match="coordinator"):
            workflow.finish_task(foreign, handoff.attempt, NodeLoopOutcome(task.id, True))
        assert graph.groups.task_result(task.id) is None
        workflow.fail_launch(owner, handoff, "spawn failed")
        assert workflow.dispatch_state(handoff.attempt.attempt_id) == "launch_failed"
        assert graph.groups.task_result(task.id) == ("failed", "spawn failed")
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.PENDING
        assert [
            event.status
            for event in control_history(conn, owner.id)
            if event.kind == "run_transition"
        ] == [
            "claimed",
            "launch_failed",
        ]
    graph.close()


def test_group_retry_repairs_link_after_persistence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        workspace = GroupWorkspace("/tmp/retry", "retry", "provider-retry")
        original = link_entity
        failed = False

        def fail_once(
            target: sqlite3.Connection, session_id: str, kind: str, entity_id: str
        ) -> None:
            nonlocal failed
            if kind == "execution_group" and not failed:
                failed = True
                raise sqlite3.OperationalError("injected link failure")
            original(target, session_id, kind, entity_id)

        monkeypatch.setattr("milknado.domains.coordinator.workflow.link_entity", fail_once)
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            _ = workflow.create_group(session, "retry-graph", (task.id,), workspace)
        group = workflow.create_group(session, "retry-graph", (task.id,), workspace)
        assert graph.groups.tasks(group.id) == (task.id,)
        assert (
            links_for_session(conn, session.id).count(EntityLink("execution_group", group.id)) == 1
        )
    graph.close()


def test_reserved_task_rejects_competing_claim(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        group = workflow.create_group(
            session, "main", (task.id,), GroupWorkspace("/tmp/claim", "claim", "provider")
        )
        handoff = workflow.dispatch_task(session, group.id, task.id, "run")
        assert not graph.claim_node(task.id, "other", now="2026-01-01T00:00:00+00:00")
        assert not graph.claim_node(
            task.id, handoff.attempt.attempt_id, now="2026-01-01T00:00:00+00:00"
        )
        with pytest.raises(ValueError):
            graph.mark_running(task.id, run_id=handoff.attempt.attempt_id)
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.PENDING
        workflow.fail_launch(session, handoff, "spawn failed")
    graph.close()


def test_review_pauses_dispatch_and_reserved_launch(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        first = graph.add_node("First", session.goal_id)
        second = graph.add_node("Second", session.goal_id)
        first_group = workflow.create_group(
            session, "first", (first.id,), GroupWorkspace("/tmp/first", "first", "provider-1")
        )
        second_group = workflow.create_group(
            session, "second", (second.id,), GroupWorkspace("/tmp/second", "second", "provider-2")
        )
        handoff = workflow.dispatch_task(session, first_group.id, first.id, "run-1")
        _ = graph.request_goal_review(
            GoalReviewRequest(
                session.goal_id, "rev", "evidence", "change", (first.id, second.id), "agent"
            )
        )
        with pytest.raises(ValueError, match="pauses"):
            _ = workflow.dispatch_task(session, second_group.id, second.id, "run-2")
        assert graph.groups.active_attempt(second_group.id) is None
        with pytest.raises(ValueError, match="pauses"):
            _ = workflow.acknowledge_launch(session, handoff)
        node = graph.get_node(first.id)
        assert node is not None and node.status is NodeStatus.PENDING
    graph.close()


def test_claim_before_reservation_is_rejected(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        assert graph.claim_node(task.id, "prior-run", now="2026-01-01T00:00:00+00:00")
        with pytest.raises(ValueError, match="active node"):
            _ = workflow.create_group(
                session, "main", (task.id,), GroupWorkspace("/tmp/prior", "prior", "provider")
            )
    graph.close()


@pytest.mark.parametrize("status", [NodeStatus.FAILED, NodeStatus.BLOCKED])
def test_launch_failure_preserves_prelaunch_state(tmp_path: Path, status: NodeStatus) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        if status is NodeStatus.FAILED:
            graph.mark_failed(task.id)
        else:
            assert graph.claim_node(task.id, "prior-run", now="2026-01-01T00:00:00+00:00")
            graph.mark_blocked(task.id)
        group = workflow.create_group(
            session, "main", (task.id,), GroupWorkspace("/tmp/retry-state", "state", "provider")
        )
        handoff = workflow.dispatch_task(session, group.id, task.id, "run")
        workflow.fail_launch(session, handoff, "spawn failed")
        node = graph.get_node(task.id)
        assert node is not None and node.status is status
        if status is NodeStatus.BLOCKED:
            prior_run = cast(
                tuple[str] | None,
                conn.execute("SELECT run_id FROM nodes WHERE id = ?", (task.id,)).fetchone(),
            )
            assert prior_run == ("prior-run",)
        assert graph.groups.task_result(task.id) == ("failed", "spawn failed")
    graph.close()
