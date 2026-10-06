from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.common import SessionContext, SessionInput
from milknado.domains.coordinator import ControlEvent, EntityLink, ProviderBinding
from milknado.domains.coordinator.commands import (
    CoordinatorAction,
    record_control_once,
    submit_coordinator_action,
)
from milknado.domains.coordinator.journal import append_control_event, control_history
from milknado.domains.coordinator.persistence import (
    bind_provider_session,
    link_entity,
    links_for_session,
)
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import GoalReviewRequest, GroupWorkspace, MikadoGraph
from milknado.loop.sessions import ProviderSessionIdentity, RuntimeSession, SessionChannel


def test_action_retry_repairs_history_without_resubmission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        bind_provider_session(
            conn, session.id, ProviderBinding("coordinator", session.id, "codex", "provider")
        )
        link_entity(conn, session.id, "provider_session", "provider")
        channel = SessionChannel()
        channel.start(SessionContext(family="codex", cwd=str(tmp_path)), ("steer",))
        incarnation = channel.capture_incarnation()
        assert incarnation is not None
        runtime = RuntimeSession(
            ProviderSessionIdentity("codex", "provider"), channel, incarnation
        )
        command = CoordinatorAction("command-1", SessionInput(action="steer", text="continue"))
        original = append_control_event
        failed = False

        def fail_once(target: sqlite3.Connection, session_id: str, event: ControlEvent) -> int:
            nonlocal failed
            if not failed:
                failed = True
                raise sqlite3.OperationalError("injected event failure")
            return original(target, session_id, event)

        monkeypatch.setattr(
            "milknado.domains.coordinator.commands.append_control_event", fail_once
        )
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            _ = submit_coordinator_action(conn, session, runtime, command)
        receipt = submit_coordinator_action(conn, session, runtime, command)
        assert receipt.state == "queued"
        assert len(channel.drain()) == 1
        assert [event.kind for event in control_history(conn, session.id)] == ["command"]
    graph.close()


def test_review_retry_repairs_link_after_persistence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        request = GoalReviewRequest(
            session.goal_id,
            "rev",
            "new evidence",
            "change goal",
            (task.id,),
            "agent",
            operation_id="review-retry",
        )
        original = link_entity
        failed = False

        def fail_once(
            target: sqlite3.Connection, session_id: str, kind: str, entity_id: str
        ) -> None:
            nonlocal failed
            if kind == "approval" and not failed:
                failed = True
                raise sqlite3.OperationalError("injected link failure")
            original(target, session_id, kind, entity_id)

        monkeypatch.setattr("milknado.domains.coordinator.workflow.link_entity", fail_once)
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            _ = workflow.review_goal_change(session, request)
        review = workflow.review_goal_change(session, request)
        assert review.goal_id == session.goal_id
        assert [link.entity_id for link in links_for_session(conn, session.id)] == [
            str(review.review_id)
        ]
        assert [event.kind for event in control_history(conn, session.id)] == ["approval"]
    graph.close()


def test_dispatch_retry_repairs_link_and_launch_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        group = workflow.create_group(
            session, "main", (task.id,), GroupWorkspace("/tmp/dispatch", "dispatch", "provider")
        )
        original_link = link_entity
        failed = False

        def fail_link(
            target: sqlite3.Connection, session_id: str, kind: str, entity_id: str
        ) -> None:
            nonlocal failed
            if kind == "run" and not failed:
                failed = True
                raise sqlite3.OperationalError("injected run link failure")
            original_link(target, session_id, kind, entity_id)

        monkeypatch.setattr("milknado.domains.coordinator.workflow.link_entity", fail_link)
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            _ = workflow.dispatch_task(session, group.id, task.id, "run")
        reserved = graph.groups.active_attempt(group.id)
        assert reserved is not None
        handoff = workflow.dispatch_task(session, group.id, task.id, "run")
        assert handoff.attempt == reserved
        assert (
            links_for_session(conn, session.id).count(EntityLink("run", reserved.attempt_id)) == 1
        )

        original_event = append_control_event
        failed = False

        def fail_event(target: sqlite3.Connection, session_id: str, event: ControlEvent) -> int:
            nonlocal failed
            if event.status == "running" and not failed:
                failed = True
                raise sqlite3.OperationalError("injected launch event failure")
            return original_event(target, session_id, event)

        monkeypatch.setattr(
            "milknado.domains.coordinator.commands.append_control_event", fail_event
        )
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            _ = workflow.acknowledge_launch(session, handoff)
        relaunched = workflow.acknowledge_launch(session, handoff)
        assert relaunched.state == "launched"
        transitions = [
            event.status
            for event in control_history(conn, session.id)
            if event.kind == "run_transition"
        ]
        assert transitions == ["claimed", "running"]
    graph.close()


def test_failed_launch_retry_cannot_change_reserved_task_outcome(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        group = workflow.create_group(
            session, "main", (task.id,), GroupWorkspace("/tmp/launch", "launch", "provider")
        )
        handoff = workflow.dispatch_task(session, group.id, task.id, "run")
        with pytest.raises(ValueError, match="another active attempt"):
            _ = workflow.dispatch_task(session, group.id, task.id, "competing-run")
        with pytest.raises(ValueError, match="launched coordinator attempt"):
            workflow.finish_task(session, handoff.attempt, NodeLoopOutcome(task.id, True))
        with pytest.raises(ValueError, match="cannot record launch failure"):
            workflow.fail_launch(session, handoff, "")
        assert graph.groups.task_result(task.id) is None

        workflow.fail_launch(session, handoff, "spawn failed")
        with pytest.raises(ValueError, match="not awaiting launch"):
            _ = workflow.acknowledge_launch(session, handoff)
        with pytest.raises(ValueError, match="conflicts with task result"):
            workflow.fail_launch(session, handoff, "different failure")
        assert workflow.dispatch_state(handoff.attempt.attempt_id) == "launch_failed"
        assert graph.groups.task_result(task.id) == ("failed", "spawn failed")
        assert [
            event.status
            for event in control_history(conn, session.id)
            if event.kind == "run_transition"
        ] == ["claimed", "launch_failed"]
    graph.close()


def test_control_event_identity_is_database_enforced(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session = CoordinatorWorkflow(graph, conn).start_goal("Goal", "codex")
        event = ControlEvent(
            kind="command", entity_kind="coordinator_command", entity_id="op-1", status="queued"
        )
        record_control_once(conn, session.id, event)
        record_control_once(conn, session.id, event)
        assert len(control_history(conn, session.id)) == 1
        with pytest.raises(sqlite3.IntegrityError):
            _ = append_control_event(conn, session.id, event)
    graph.close()
