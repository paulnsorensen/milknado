from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from milknado.domains.common import NodeKind, NodeSpec, RunResult
from milknado.domains.coordinator.commands import (
    DispatchHandoff,
    create_dispatch_table,
    get_dispatch_state,
    owned_dispatch_state,
    record_control_once,
)
from milknado.domains.coordinator.model import ControlEvent, CoordinatorSession
from milknado.domains.coordinator.persistence import link_entity, start_coordinator
from milknado.domains.coordinator.planning_workflow import CoordinatorPlanning
from milknado.domains.coordinator.plans import PlanProposalRecord
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
from milknado.domains.planning import Planner


class CoordinatorWorkflow:  # noqa: V102
    def __init__(self, graph: MikadoGraph, conn: sqlite3.Connection) -> None:
        self._graph: MikadoGraph = graph
        self._conn: sqlite3.Connection = conn
        self._planning: CoordinatorPlanning = CoordinatorPlanning(graph, conn)

    def _event_once(self, session: CoordinatorSession, event: ControlEvent) -> None:
        record_control_once(self._conn, session.id, event)

    def _owned_group(
        self, session: CoordinatorSession, group_id: str, node_id: int
    ) -> ExecutionGroup:
        group = self._graph.groups.get(group_id)
        admission = self._graph.goal_admission(node_id)
        linked = cast(
            tuple[int] | None,
            self._conn.execute(
                "SELECT 1 FROM coordinator_links WHERE session_id = ? "
                + "AND kind = 'execution_group' AND entity_id = ?",
                (session.id, group_id),
            ).fetchone(),
        )
        if (
            group is None
            or node_id not in self._graph.groups.tasks(group_id)
            or admission.goal_id != session.goal_id
            or linked is None
        ):
            raise ValueError("task or group belongs to another coordinator")
        return group

    def _owned_attempt(self, session: CoordinatorSession, attempt: TaskAttempt) -> str:
        _ = self._owned_group(session, attempt.group_id, attempt.node_id)
        return owned_dispatch_state(self._conn, session.id, attempt)

    def start_goal(self, description: str, provider: str) -> CoordinatorSession:  # noqa: V105
        if not description.strip():
            raise ValueError("goal description must not be empty")
        goal = self._graph.add_node(description, spec=NodeSpec(kind=NodeKind.GOAL))
        return start_coordinator(self._conn, goal.id, provider)

    def plan_goal(
        self, session: CoordinatorSession, planner: Planner, project_root: Path, operation_id: str
    ) -> PlanProposalRecord:
        return self._planning.plan_goal(session, planner, project_root, operation_id)

    def decide_plan(  # noqa: PLR0913 - approval needs session, planner, root, and decision
        self,
        session: CoordinatorSession,
        planner: Planner,
        project_root: Path,
        proposal_id: str,
        decision: str,
    ) -> PlanProposalRecord:
        return self._planning.decide_plan(session, planner, project_root, proposal_id, decision)

    def record_plan(self, session: CoordinatorSession, plan_id: str, status: str) -> None:
        self._planning.record_plan(session, plan_id, status)

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
        existing = self._graph.groups.for_task(tasks[0]) if tasks else None
        if existing is None:
            group = self._graph.groups.create(graph_id, tasks, workspace)
        else:
            if (
                self._graph.groups.tasks(existing.id) != tasks
                or existing.graph_id != graph_id
                or (existing.worktree_path, existing.branch_name, existing.provider_session_id)
                != (workspace.worktree_path, workspace.branch_name, workspace.provider_session_id)
            ):
                raise ValueError("task belongs to a different execution group")
            group = existing
        link_entity(self._conn, session.id, "execution_group", group.id)
        if group.provider_session_id is not None:
            link_entity(self._conn, session.id, "provider_session", group.provider_session_id)
        self._event_once(
            session,
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
    ) -> DispatchHandoff:
        _ = self._owned_group(session, group_id, node_id)
        if not self._graph.goal_admission(node_id).allowed:
            raise ValueError("goal review pauses task dispatch")
        create_dispatch_table(self._conn)
        attempt = self._graph.groups.active_attempt(group_id)
        if attempt is None:
            attempt = self._graph.groups.reserve_task(group_id, node_id, run_id)
        elif (attempt.node_id, attempt.run_id) != (node_id, run_id):
            raise ValueError("execution group has another active attempt")
        with self._conn:
            _ = self._conn.execute(
                "INSERT OR IGNORE INTO coordinator_dispatches "
                + "(attempt_id, session_id, group_id, node_id, run_id, state) "
                + "VALUES (?, ?, ?, ?, ?, 'awaiting_launch')",
                (attempt.attempt_id, session.id, group_id, node_id, run_id),
            )
        state = self._owned_attempt(session, attempt)
        link_entity(self._conn, session.id, "run", attempt.attempt_id)
        self._event_once(
            session,
            ControlEvent(
                "run_transition", entity_kind="run", entity_id=attempt.attempt_id, status="claimed"
            ),
        )
        return DispatchHandoff(attempt, state)

    def dispatch_state(self, attempt_id: str) -> str | None:  # noqa: V105
        return get_dispatch_state(self._conn, attempt_id)

    def acknowledge_launch(  # noqa: V105
        self, session: CoordinatorSession, handoff: DispatchHandoff
    ) -> DispatchHandoff:
        state = self._owned_attempt(session, handoff.attempt)
        if state not in {"awaiting_launch", "launched"}:
            raise ValueError("dispatch is not awaiting launch")
        self._graph.groups.launch_reserved_task(handoff.attempt)
        run = self._graph.runs.get(handoff.attempt.attempt_id)
        group = self._graph.groups.get(handoff.attempt.group_id)
        if group is None:
            raise ValueError("execution group disappeared during launch")
        if run is None:
            self._graph.runs.start(
                handoff.attempt.attempt_id,
                handoff.attempt.node_id,
                group.worktree_path,
                datetime.now(UTC).isoformat(),
                None,
            )
        elif run["node_id"] != handoff.attempt.node_id or run["status"] != "running":
            raise ValueError("group run identity conflicts with launched writer")
        with self._conn:
            _ = self._conn.execute(
                "UPDATE coordinator_dispatches SET state = 'launched' WHERE attempt_id = ?",
                (handoff.attempt.attempt_id,),
            )
        self._event_once(
            session,
            ControlEvent(
                kind="run_transition",
                entity_kind="run",
                entity_id=handoff.attempt.attempt_id,
                status="running",
            ),
        )
        return DispatchHandoff(handoff.attempt, "launched")

    def fail_launch(  # noqa: V105
        self, session: CoordinatorSession, handoff: DispatchHandoff, reason: str
    ) -> None:
        state = self._owned_attempt(session, handoff.attempt)
        if state not in {"awaiting_launch", "launch_failed"} or not reason:
            raise ValueError("dispatch cannot record launch failure")
        result = self._graph.groups.task_result(handoff.attempt.node_id)
        if result is None:
            self._graph.groups.fail_reserved_task(handoff.attempt, reason)
        elif result != ("failed", reason):
            raise ValueError("launch failure conflicts with task result")
        with self._conn:
            _ = self._conn.execute(
                "UPDATE coordinator_dispatches SET state = 'launch_failed' WHERE attempt_id = ?",
                (handoff.attempt.attempt_id,),
            )
        self._event_once(
            session,
            ControlEvent(
                kind="run_transition",
                entity_kind="run",
                entity_id=handoff.attempt.attempt_id,
                status="launch_failed",
            ),
        )

    def finish_task(
        self, session: CoordinatorSession, attempt: TaskAttempt, outcome: NodeLoopOutcome
    ) -> None:
        state = self._owned_attempt(session, attempt)
        if state not in {"launched", "finished"} or outcome.node_id != attempt.node_id:
            raise ValueError("worker result does not match a launched coordinator attempt")
        status = (
            "done" if outcome.success else "blocked" if outcome.ownership_preserved else "failed"
        )
        expected = (status, outcome.detail or "")
        result = self._graph.groups.task_result(attempt.node_id)
        if result is None:
            self._graph.groups.finish_task(attempt, TaskOutcome(*expected))
        elif result != expected:
            raise ValueError("worker result conflicts with recorded task result")
        run = self._graph.runs.get(attempt.attempt_id)
        if run is not None and run["status"] == "running":
            self._graph.runs.finish(
                attempt.attempt_id,
                RunResult(
                    status, None, False, datetime.now(UTC).isoformat(), detail=outcome.detail
                ),
            )
        with self._conn:
            _ = self._conn.execute(
                "UPDATE coordinator_dispatches SET state = 'finished' WHERE attempt_id = ?",
                (attempt.attempt_id,),
            )
        self._event_once(
            session,
            ControlEvent(
                "run_transition", entity_kind="run", entity_id=attempt.attempt_id, status=status
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
        self._event_once(
            session,
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
        if not request.operation_id:
            raise ValueError("goal review operation identity must not be empty")
        review = self._graph.request_goal_review(request)
        link_entity(self._conn, session.id, "approval", str(review.review_id))
        self._event_once(
            session,
            ControlEvent(
                kind="approval",
                entity_kind="goal_review",
                entity_id=str(review.review_id),
                status=review.decision.value,
            ),
        )
        return review
