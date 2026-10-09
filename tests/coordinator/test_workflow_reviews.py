from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.common import CONTROLLER_MASTER_ENV
from milknado.domains.coordinator.journal import control_history
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import (
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRequest,
    MikadoGraph,
)


def test_goal_review_retry_matches_canonical_text(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        original = GoalReviewRequest(
            session.goal_id,
            "rev",
            "evidence",
            "change",
            (task.id,),
            "agent",
            operation_id="review-whitespace",
        )
        first = workflow.review_goal_change(session, original)
        spaced = GoalReviewRequest(
            session.goal_id,
            " rev ",
            " evidence ",
            " change ",
            (task.id,),
            " agent ",
            operation_id="review-whitespace",
        )
        retried = workflow.review_goal_change(session, spaced)
        assert retried.review_id == first.review_id
        approvals = [
            event.entity_id
            for event in control_history(conn, session.id)
            if event.kind == "approval"
        ]
        assert approvals == [str(first.review_id)]
    graph.close()


@pytest.mark.parametrize("decision", [GoalReviewDecision.ACCEPTED, GoalReviewDecision.REJECTED])
def test_decided_review_retry_keeps_original_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: GoalReviewDecision
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "controller-secret")
    graph.register_controller_master()
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        session = workflow.start_goal("Goal", "codex")
        task = graph.add_node("Task", session.goal_id)
        request = GoalReviewRequest(
            session.goal_id,
            "rev",
            "evidence",
            "change",
            (task.id,),
            "agent",
            operation_id=f"review-{decision.value}",
        )
        first = workflow.review_goal_change(session, request)
        _ = graph.decide_goal_review(
            GoalReviewDecisionRequest(first.review_id, decision), decided_by="human"
        )
        retried = workflow.review_goal_change(session, request)
        assert retried.review_id == first.review_id
        assert retried.decision is decision
        latest = graph.latest_goal_review(session.goal_id)
        assert latest is not None and latest.review_id == first.review_id
    graph.close()
