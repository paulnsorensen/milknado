from __future__ import annotations

import sqlite3
from pathlib import Path

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.coordinator.journal import append_control_event
from milknado.domains.coordinator.model import ControlEvent, CoordinatorSession
from milknado.domains.coordinator.persistence import link_entity, start_coordinator
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import (
    ExecutionGroup,
    GoalReviewRecord,
    GoalReviewRequest,
    GroupWorkspace,
    MikadoGraph,
    TaskAttempt,
    TaskOutcome,
)
from milknado.domains.planning import Planner, PlanResult


class CoordinatorWorkflow:
    def __init__(self, graph: MikadoGraph, conn: sqlite3.Connection) -> None:
        self._graph: MikadoGraph = graph
        self._conn: sqlite3.Connection = conn

    def start_goal(self, description: str, provider: str) -> CoordinatorSession:  # noqa: V105
        if not description.strip():
            raise ValueError("goal description must not be empty")
        goal = self._graph.add_node(description, spec=NodeSpec(kind=NodeKind.GOAL))
        return start_coordinator(self._conn, goal.id, provider)

    def plan_goal(  # noqa: V105
        self, session: CoordinatorSession, planner: Planner, project_root: Path
    ) -> PlanResult:
        goal = self._graph.get_node(session.goal_id)
        if goal is None:
            raise ValueError("coordinator goal does not exist")
        result = planner.launch(goal.description, project_root)
        self.record_plan(
            session, str(result.context_path), "accepted" if result.success else "failed"
        )
        return result

    def record_plan(self, session: CoordinatorSession, plan_id: str, status: str) -> None:
        if not plan_id:
            raise ValueError("plan identity must not be empty")
        link_entity(self._conn, session.id, "planning_decision", plan_id)
        _ = append_control_event(
            self._conn,
            session.id,
            ControlEvent(
                kind="planning_decision", entity_kind="plan", entity_id=plan_id, status=status
            ),
        )

    def create_group(  # noqa: V105
        self,
        session: CoordinatorSession,
        graph_id: str,
        tasks: tuple[int, ...],
        workspace: GroupWorkspace,
    ) -> ExecutionGroup:
        for node_id in tasks:
            if self._graph.goal_admission(node_id).goal_id != session.goal_id:
                raise ValueError("execution group task belongs to another goal")
        group = self._graph.groups.create(graph_id, tasks, workspace)
        link_entity(self._conn, session.id, "execution_group", group.id)
        link_entity(self._conn, session.id, "provider_session", group.provider_session_id)
        _ = append_control_event(
            self._conn,
            session.id,
            ControlEvent(
                kind="execution_group",
                entity_kind="execution_group",
                entity_id=group.id,
                status="created",
            ),
        )
        return group

    def dispatch_task(  # noqa: V105
        self, session: CoordinatorSession, group_id: str, node_id: int, run_id: str
    ) -> TaskAttempt:
        if self._graph.goal_admission(node_id).goal_id != session.goal_id:
            raise ValueError("task belongs to another coordinator goal")
        attempt = self._graph.groups.start_task(group_id, node_id, run_id)
        link_entity(self._conn, session.id, "run", run_id)
        _ = append_control_event(
            self._conn,
            session.id,
            ControlEvent(
                kind="run_transition", entity_kind="run", entity_id=run_id, status="running"
            ),
        )
        return attempt

    def finish_task(
        self, session: CoordinatorSession, attempt: TaskAttempt, outcome: NodeLoopOutcome
    ) -> None:
        if outcome.node_id != attempt.node_id:
            raise ValueError("worker result addresses another task")
        status = (
            "done" if outcome.success else "blocked" if outcome.ownership_preserved else "failed"
        )
        self._graph.groups.finish_task(attempt, TaskOutcome(status, outcome.detail or ""))
        _ = append_control_event(
            self._conn,
            session.id,
            ControlEvent(
                kind="run_transition", entity_kind="run", entity_id=attempt.run_id, status=status
            ),
        )

    def record_revision(  # noqa: V105
        self, session: CoordinatorSession, revision_id: str, affected_node_ids: tuple[int, ...]
    ) -> None:
        if not revision_id or not affected_node_ids:
            raise ValueError("revision identity and affected tasks are required")
        for node_id in affected_node_ids:
            admission = self._graph.goal_admission(node_id)
            if not admission.allowed or admission.goal_id != session.goal_id:
                raise ValueError("revision task is not admitted under this goal")
        link_entity(self._conn, session.id, "graph_revision", revision_id)
        _ = append_control_event(
            self._conn,
            session.id,
            ControlEvent(
                kind="graph_revision",
                entity_kind="graph_revision",
                entity_id=revision_id,
                status="applied",
            ),
        )

    def review_goal_change(  # noqa: V105
        self, session: CoordinatorSession, request: GoalReviewRequest
    ) -> GoalReviewRecord:
        if request.goal_id != session.goal_id:
            raise ValueError("review addresses another coordinator goal")
        review = self._graph.request_goal_review(request)
        link_entity(self._conn, session.id, "approval", str(review.review_id))
        _ = append_control_event(
            self._conn,
            session.id,
            ControlEvent(
                kind="approval",
                entity_kind="goal_review",
                entity_id=str(review.review_id),
                status="pending",
            ),
        )
        return review
