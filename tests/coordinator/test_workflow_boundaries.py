from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.coordinator.journal import control_history
from milknado.domains.coordinator.persistence import links_for_session
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import GoalReviewRequest, GroupWorkspace, MikadoGraph


def test_empty_goal_and_plan_do_not_write_coordinator_state(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        with pytest.raises(ValueError, match="goal description"):
            _ = workflow.start_goal("  ", "codex")
        assert graph.get_all_nodes() == []
        session = workflow.start_goal("Goal", "codex")
        with pytest.raises(ValueError, match="plan identity"):
            workflow.record_plan(session, "", "accepted")
        assert links_for_session(conn, session.id) == ()
        assert control_history(conn, session.id) == ()
    graph.close()


def test_foreign_group_task_and_incompatible_retry_keep_original_group(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        owner = workflow.start_goal("Owner", "codex")
        foreign = workflow.start_goal("Foreign", "codex")
        own_task = graph.add_node("Own task", owner.goal_id)
        foreign_task = graph.add_node("Foreign task", foreign.goal_id)
        workspace = GroupWorkspace(str(tmp_path / "group"), "branch", "provider")
        before_links = links_for_session(conn, owner.id)
        before_events = control_history(conn, owner.id)
        with pytest.raises(ValueError, match="another goal"):
            _ = workflow.create_group(owner, "main", (foreign_task.id,), workspace)
        assert graph.groups.for_task(foreign_task.id) is None
        assert links_for_session(conn, owner.id) == before_links
        assert control_history(conn, owner.id) == before_events
        group = workflow.create_group(owner, "main", (own_task.id,), workspace)
        prior_links = links_for_session(conn, owner.id)
        prior_events = control_history(conn, owner.id)
        with pytest.raises(ValueError, match="different execution group"):
            _ = workflow.create_group(
                owner,
                "other",
                (own_task.id,),
                GroupWorkspace(str(tmp_path / "other"), "other", "provider-2"),
            )
        assert graph.groups.for_task(own_task.id) == group
        assert links_for_session(conn, owner.id) == prior_links
        assert control_history(conn, owner.id) == prior_events
    graph.close()


def test_conflicting_finished_result_keeps_task_and_journal(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        group = workflow.create_group(
            session, "main", (task.id,), GroupWorkspace(str(tmp_path), "branch", "provider")
        )
        handoff = workflow.dispatch_task(session, group.id, task.id, "run-1")
        _ = workflow.acknowledge_launch(session, handoff)
        workflow.finish_task(session, handoff.attempt, NodeLoopOutcome(task.id, True, "first"))
        prior_links = links_for_session(conn, session.id)
        prior_events = control_history(conn, session.id)
        with pytest.raises(ValueError, match="conflicts"):
            workflow.finish_task(
                session, handoff.attempt, NodeLoopOutcome(task.id, False, "second")
            )
        assert graph.groups.task_result(task.id) == ("done", "first")
        assert workflow.dispatch_state(handoff.attempt.attempt_id) == "finished"
        assert links_for_session(conn, session.id) == prior_links
        assert control_history(conn, session.id) == prior_events
    graph.close()


def test_invalid_revision_and_goal_review_keep_durable_state(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        owner = workflow.start_goal("Owner", "codex")
        foreign = workflow.start_goal("Foreign", "codex")
        own_task = graph.add_node("Own task", owner.goal_id)
        foreign_task = graph.add_node("Foreign task", foreign.goal_id)
        before_links = links_for_session(conn, owner.id)
        before_events = control_history(conn, owner.id)
        for revision, nodes in (("", (own_task.id,)), ("rev", ()), ("rev", (foreign_task.id,))):
            with pytest.raises(ValueError, match="revision"):
                workflow.record_revision(owner, revision, nodes)
        before_reviews = conn.execute(
            "SELECT review_id, goal_id, operation_id FROM goal_reviews ORDER BY review_id"
        ).fetchall()
        for goal_id, operation_id, message in (
            (foreign.goal_id, "review-1", "another coordinator"),
            (owner.goal_id, "", "operation identity"),
        ):
            request = GoalReviewRequest(
                goal_id,
                "rev",
                "evidence",
                "change",
                (own_task.id,),
                "agent",
                operation_id=operation_id,
            )
            with pytest.raises(ValueError, match=message):
                _ = workflow.review_goal_change(owner, request)
            assert (
                conn.execute(
                    "SELECT review_id, goal_id, operation_id FROM goal_reviews ORDER BY review_id"
                ).fetchall()
                == before_reviews
            )
        assert links_for_session(conn, owner.id) == before_links
        assert control_history(conn, owner.id) == before_events
        assert graph.goal_admission(own_task.id).allowed
    graph.close()
