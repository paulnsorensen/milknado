from __future__ import annotations

import sqlite3
from pathlib import Path

from milknado.domains.coordinator.commands import record_control_once
from milknado.domains.coordinator.model import ControlEvent, CoordinatorSession
from milknado.domains.coordinator.persistence import link_entity
from milknado.domains.coordinator.plans import (
    PlanProposalRecord,
    get_proposal,
    save_proposal,
    transition_proposal,
)
from milknado.domains.graph import MikadoGraph, graph_revision
from milknado.domains.planning import Planner, record_batch_snapshot


class CoordinatorPlanning:
    def __init__(self, graph: MikadoGraph, conn: sqlite3.Connection) -> None:
        self._graph: MikadoGraph = graph
        self._conn: sqlite3.Connection = conn

    def plan_goal(
        self, session: CoordinatorSession, planner: Planner, project_root: Path, operation_id: str
    ) -> PlanProposalRecord:
        with self._graph.synchronization_lock:
            try:
                existing = get_proposal(self._conn, operation_id)
            except KeyError:
                pass
            else:
                if existing.session_id != session.id:
                    raise ValueError("plan proposal belongs to another coordinator")
                return existing
            revision = graph_revision(self._conn)
            goal = self._graph.get_node(session.goal_id)
            if goal is None:
                raise ValueError("coordinator goal does not exist")
        proposal = planner.propose(goal.description, project_root, target_goal_id=session.goal_id)
        record = PlanProposalRecord(
            operation_id,
            session.id,
            proposal.manifest,
            str(proposal.context_path),
            revision,
            "pending",
        )
        with self._graph.synchronization_lock:
            return save_proposal(self._conn, record)

    def decide_plan(  # noqa: PLR0913 - approval needs session, planner, root, and decision
        self,
        session: CoordinatorSession,
        planner: Planner,
        project_root: Path,
        proposal_id: str,
        decision: str,
    ) -> PlanProposalRecord:
        with self._graph.synchronization_lock:
            record = self._preflight_plan(session, proposal_id, decision)
            if record.status in {"applied", "rejected"}:
                return record
        proposal = record.proposal()
        prepared_plan = planner.prepare_proposal(proposal, project_root)
        with self._graph.synchronization_lock:
            _ = transition_proposal(self._conn, proposal_id, "pending", "applying")
            with self._graph.plan_transaction(record.graph_revision) as current:
                if current:
                    result = planner.apply_proposal(
                        proposal,
                        target_goal_id=session.goal_id,
                        prepared_plan=prepared_plan,
                    )
                    if not result.success:
                        raise RuntimeError(
                            "plan apply did not complete; manual recovery is required"
                        )
            if not current:
                _ = transition_proposal(self._conn, proposal_id, "applying", "stale")
                raise ValueError("plan proposal is stale; request a new proposal")
        record_batch_snapshot(project_root, record.manifest, prepared_plan)
        with self._graph.synchronization_lock:
            applied = transition_proposal(self._conn, proposal_id, "applying", "applied")
            self.record_plan(session, proposal_id, "accepted")
            return applied

    def _preflight_plan(
        self, session: CoordinatorSession, proposal_id: str, decision: str
    ) -> PlanProposalRecord:
        record = get_proposal(self._conn, proposal_id)
        if record.session_id != session.id:
            raise ValueError("plan proposal belongs to another coordinator")
        if record.status == "applying":
            raise ValueError("plan apply is incomplete; manual recovery is required")
        if record.status == "stale":
            raise ValueError("plan proposal is stale; request a new proposal")
        if record.status in {"applied", "rejected"}:
            if (record.status == "applied") == (decision == "accepted"):
                if record.status == "applied":
                    self.record_plan(session, proposal_id, "accepted")
                return record
            raise ValueError("plan proposal was already decided")
        if decision == "rejected":
            return transition_proposal(self._conn, proposal_id, "pending", "rejected")
        if graph_revision(self._conn) != record.graph_revision:
            _ = transition_proposal(self._conn, proposal_id, "pending", "stale")
            raise ValueError("plan proposal is stale; request a new proposal")
        return record

    def record_plan(self, session: CoordinatorSession, plan_id: str, status: str) -> None:
        if not plan_id:
            raise ValueError("plan identity must not be empty")
        with self._graph.synchronization_lock:
            link_entity(self._conn, session.id, "planning_decision", plan_id)
            record_control_once(
                self._conn,
                session.id,
                ControlEvent(
                    kind="planning_decision", entity_kind="plan", entity_id=plan_id, status=status
                ),
            )
