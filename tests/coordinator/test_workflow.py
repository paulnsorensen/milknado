from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import SessionContext, SessionEvent, SessionInput
from milknado.domains.coordinator import (
    CoordinatorAction,
    CoordinatorWorkflow,
    submit_coordinator_action,
)
from milknado.domains.coordinator.journal import control_history
from milknado.domains.coordinator.persistence import links_for_session
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import GoalReviewRequest, GroupWorkspace, MikadoGraph
from milknado.loop.sessions import ProviderSessionIdentity, RuntimeSession, SessionChannel


def test_goal_plan_group_and_receipts_survive_reopen(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Deliver result", "codex")
        task = graph.add_node("Implement", session.goal_id, files=("src/a.py",))
        workflow.record_plan(session, "plan-1", "accepted")
        group = workflow.create_group(
            session, "main", (task.id,), GroupWorkspace("/tmp/group-a", "group-a", "provider-a")
        )
        attempt = workflow.dispatch_task(session, group.id, task.id, "run-a")
        workflow.finish_task(session, attempt, NodeLoopOutcome(task.id, True, "verified"))
        assert graph.groups.task_result(task.id) == ("done", "verified")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        links = links_for_session(conn, session.id)
        assert [(link.kind, link.entity_id) for link in links] == [
            ("planning_decision", "plan-1"),
            ("execution_group", group.id),
            ("provider_session", "provider-a"),
            ("run", "run-a"),
        ]
        assert [event.kind for event in control_history(conn, session.id)] == [
            "planning_decision",
            "execution_group",
            "run_transition",
            "run_transition",
        ]
    graph.close()


def test_revision_policy_pauses_only_affected_work(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Deliver result", "codex")
        affected = graph.add_node("Affected", session.goal_id)
        other = graph.add_node("Other", session.goal_id)
        workflow.record_revision(session, "rev-1", (affected.id,))
        assert graph.goal_admission(affected.id).allowed
        review = workflow.review_goal_change(
            session,
            GoalReviewRequest(
                session.goal_id, "rev-1", "new evidence", "change outcome", (affected.id,), "agent"
            ),
        )
        assert review.affected_node_ids == (affected.id,)
        assert not graph.goal_admission(affected.id).allowed
        assert graph.goal_admission(other.id).allowed
        assert review.interruption_receipts == ()
    graph.close()


def test_unbounded_goal_change_pauses_all_work(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Deliver result", "claude")
        first = graph.add_node("First", session.goal_id)
        second = graph.add_node("Second", session.goal_id)
        review = workflow.review_goal_change(
            session,
            GoalReviewRequest(
                session.goal_id, "rev-2", "unknown impact", "change outcome", None, "agent"
            ),
        )
        assert review.unbounded
        assert not graph.goal_admission(first.id).allowed
        assert not graph.goal_admission(second.id).allowed
    graph.close()


def test_approval_action_is_queued_once_with_durable_receipt(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Deliver result", "codex")
        channel = SessionChannel()
        channel.start(SessionContext(family="codex", cwd=str(tmp_path)), ("approve",))
        channel.publish(
            SessionEvent(kind="permission", text="write?", event_id="p-1", state="requested")
        )
        incarnation = channel.capture_incarnation()
        assert incarnation is not None
        runtime = RuntimeSession(
            ProviderSessionIdentity("codex", "provider-1"), channel, incarnation
        )
        action = SessionInput(action="approve", request_id=channel.view().permissions[0].event_id)
        command = CoordinatorAction("command-1", action)
        first = submit_coordinator_action(conn, session, runtime, command)
        second = submit_coordinator_action(conn, session, runtime, command)
        assert first == second
        assert first.state == "queued"
        assert len(channel.drain()) == 1
        with pytest.raises(ValueError, match="reused"):
            _ = submit_coordinator_action(
                conn,
                session,
                runtime,
                CoordinatorAction(
                    "command-1", SessionInput(action="deny", request_id=action.request_id)
                ),
            )
    with closing(sqlite3.connect(graph.db_path)) as conn:
        row = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT state FROM coordinator_action_receipts WHERE command_id = 'command-1'"
            ).fetchone(),
        )
        assert row == ("queued",)
        assert [event.kind for event in control_history(conn, session.id)] == ["approval"]
    graph.close()
