from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import (
    CrgPort,
    NodeKind,
    NodeStatus,
    SessionContext,
    SessionEvent,
    SessionInput,
)
from milknado.domains.coordinator import (
    CoordinatorAction,
    CoordinatorWorkflow,
    EntityLink,
    submit_coordinator_action,
)
from milknado.domains.coordinator.journal import control_history
from milknado.domains.coordinator.persistence import link_entity, links_for_session
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import GoalReviewRequest, GroupWorkspace, MikadoGraph
from milknado.domains.planning import Planner, PlanningPorts, PlanningProcessResult
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
        handoff = workflow.dispatch_task(session, group.id, task.id, "run-a")
        assert handoff.state == "awaiting_launch"
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.PENDING
        _ = workflow.acknowledge_launch(session, handoff)
        node = graph.get_node(task.id)
        assert node is not None and node.status is NodeStatus.RUNNING
        workflow.finish_task(session, handoff.attempt, NodeLoopOutcome(task.id, True, "verified"))
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
        link_entity(conn, session.id, "provider_session", "provider-1")
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


class _UnavailableCrg:
    def ensure_graph(self, _project_root: Path) -> None:
        raise RuntimeError("unavailable")


class _PlanningProcess:
    def run_agent(
        self, context_path: Path, command: str, project_root: Path
    ) -> PlanningProcessResult:
        _ = (context_path, command, project_root)
        payload = {
            "manifest_version": "milknado.plan.v2",
            "goal": "Second goal",
            "goal_summary": "Second goal",
            "changes": [{"id": "c1", "path": "src/second.py", "description": "Second task"}],
        }
        return PlanningProcessResult(0, "```json\n" + json.dumps(payload) + "\n```")

    def run_validation(
        self, command: str, payload: dict[str, object], project_root: Path
    ) -> PlanningProcessResult:
        _ = (command, payload, project_root)
        return PlanningProcessResult(0)


def test_planner_attaches_manifest_to_reviewed_goal_not_first_root(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        first = workflow.start_goal("First goal", "codex")
        second = workflow.start_goal("Second goal", "codex")
        planner = Planner(
            graph,
            cast(CrgPort, cast(object, _UnavailableCrg())),
            "codex",
            PlanningPorts(_PlanningProcess()),
        )
        result = workflow.plan_goal(second, planner, tmp_path)
        assert result.success
        task = next(node for node in graph.get_all_nodes() if node.kind is NodeKind.TASK)
        assert task.parent_id == second.goal_id
        assert task.parent_id != first.goal_id
    graph.close()


def test_action_rejects_same_family_foreign_session(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        owner = workflow.start_goal("Owner", "codex")
        foreign = workflow.start_goal("Foreign", "codex")
        link_entity(conn, owner.id, "provider_session", "provider-owner")
        channel = SessionChannel()
        channel.start(SessionContext(family="codex", cwd=str(tmp_path)), ("steer",))
        incarnation = channel.capture_incarnation()
        assert incarnation is not None
        runtime = RuntimeSession(
            ProviderSessionIdentity("codex", "provider-owner"), channel, incarnation
        )
        command = CoordinatorAction("foreign-action", SessionInput(action="steer", text="change"))
        with pytest.raises(ValueError, match="provider session"):
            _ = submit_coordinator_action(conn, foreign, runtime, command)
        assert channel.drain() == ()
    graph.close()


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
        assert node is not None and node.status is NodeStatus.FAILED
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
