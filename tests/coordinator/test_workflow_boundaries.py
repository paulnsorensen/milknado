from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.coordinator.journal import control_history
from milknado.domains.coordinator.persistence import links_for_session
from milknado.domains.coordinator.planning_workflow import CoordinatorPlanning
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import GoalReviewRequest, GroupWorkspace, MikadoGraph


def test_group_and_revision_rejections_preserve_links_and_events(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    conn = graph.group_connection
    workflow = CoordinatorWorkflow(graph, conn)
    owner = workflow.start_goal("Owner", "codex")
    other = workflow.start_goal("Other", "codex")
    foreign = graph.add_node("Foreign task", other.goal_id)
    local = graph.add_node("Local task", owner.goal_id)
    workspace = GroupWorkspace(str(tmp_path / "group"), "branch", "provider")
    links = conn.execute("SELECT * FROM coordinator_links").fetchall()
    events = conn.execute("SELECT * FROM coordinator_events").fetchall()

    with pytest.raises(ValueError, match="another goal"):
        _ = workflow.create_group(owner, "main", (foreign.id,), workspace)
    with pytest.raises(ValueError, match="revision identity"):
        workflow.record_revision(owner, "", (local.id,))
    with pytest.raises(ValueError, match="not admitted"):
        workflow.record_revision(owner, "revision", (foreign.id,))
    with pytest.raises(ValueError, match="another coordinator goal"):
        _ = workflow.review_goal_change(
            owner,
            GoalReviewRequest(
                other.goal_id,
                "revision",
                "evidence",
                "change",
                (foreign.id,),
                "agent",
                operation_id="review",
            ),
        )
    assert conn.execute("SELECT * FROM coordinator_links").fetchall() == links
    assert conn.execute("SELECT * FROM coordinator_events").fetchall() == events
    assert conn.execute("SELECT COUNT(*) FROM execution_groups").fetchone()[0] == 0
    graph.close()


def test_conflicting_finish_preserves_task_result_and_dispatch(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    conn = graph.group_connection
    workflow = CoordinatorWorkflow(graph, conn)
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
    workflow.finish_task(session, handoff.attempt, NodeLoopOutcome(task.id, True, "verified"))
    before = conn.execute("SELECT * FROM coordinator_events").fetchall()
    dispatch = cast(
        tuple[str] | None,
        conn.execute(
            "SELECT state FROM coordinator_dispatches WHERE attempt_id = ?",
            (handoff.attempt.attempt_id,),
        ).fetchone(),
    )

    with pytest.raises(ValueError, match="conflicts"):
        workflow.finish_task(session, handoff.attempt, NodeLoopOutcome(task.id, False, "wrong"))
    assert graph.groups.task_result(task.id) == ("done", "verified")
    assert conn.execute("SELECT * FROM coordinator_events").fetchall() == before
    unchanged = cast(
        tuple[str] | None,
        conn.execute(
            "SELECT state FROM coordinator_dispatches WHERE attempt_id = ?",
            (handoff.attempt.attempt_id,),
        ).fetchone(),
    )
    assert unchanged == dispatch
    graph.close()


def test_empty_goal_and_plan_do_not_write_coordinator_state(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        with pytest.raises(ValueError, match="goal description"):
            _ = workflow.start_goal("  ", "codex")
        assert graph.get_all_nodes() == []
        session = workflow.start_goal("Goal", "codex")
        with pytest.raises(ValueError, match="plan identity"):
            CoordinatorPlanning(graph, conn).record_plan(session, "", "accepted")
        assert links_for_session(conn, session.id) == ()
        assert control_history(conn, session.id) == ()
    graph.close()


@pytest.mark.parametrize("provider", ("", " \t "))
def test_empty_provider_does_not_create_goal_or_coordinator(tmp_path: Path, provider: str) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        with pytest.raises(ValueError, match="goal description"):
            _ = workflow.start_goal(" ", provider)
        with pytest.raises(ValueError, match="provider must not be empty"):
            _ = workflow.start_goal("Goal", provider)
        assert graph.get_all_nodes() == []
        assert conn.execute("SELECT COUNT(*) FROM coordinator_sessions").fetchone() == (0,)
    graph.close()


def test_start_goal_preserves_nonempty_provider(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        provider = " custom/provider "
        session = CoordinatorWorkflow(graph, conn).start_goal("Goal", provider)
        assert session.provider == provider
        assert conn.execute("SELECT provider FROM coordinator_sessions").fetchone() == (provider,)
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
