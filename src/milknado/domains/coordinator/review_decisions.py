from __future__ import annotations

from typing import Literal, cast

from milknado.domains.coordinator.commands import record_control_once
from milknado.domains.coordinator.control_models import DecideGoalReview
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.coordinator.model import ControlEvent, CoordinatorSession
from milknado.domains.graph import GoalReviewDecisionRequest, GoalReviewRecord, MikadoGraph


def decide_goal_review(
    graph: MikadoGraph,
    services: CoordinatorServices,
    request: GoalReviewDecisionRequest,
    decided_by: str,
) -> GoalReviewRecord:
    if services.review_decision is None:
        raise PermissionError("Review decisions are unavailable.")
    with graph.synchronization_lock:
        conn = graph.group_connection
        linked = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT session_id FROM coordinator_links "
                + "WHERE kind = 'approval' AND entity_id = ?",
                (str(request.review_id),),
            ).fetchone(),
        )
        review = services.review_decision(request, decided_by=decided_by)
        if linked is not None:
            record_control_once(
                conn,
                linked[0],
                ControlEvent(
                    kind="approval",
                    entity_kind="goal_review",
                    entity_id=str(review.review_id),
                    status=review.decision.value,
                ),
            )
        return review


def decide_coordinator_review(
    graph: MikadoGraph,
    services: CoordinatorServices,
    session: CoordinatorSession,
    command: DecideGoalReview,
) -> tuple[Literal["accepted", "unavailable"], object]:
    linked = cast(
        tuple[int] | None,
        graph.group_connection.execute(
            "SELECT 1 FROM coordinator_links WHERE session_id = ? "
            + "AND kind = 'approval' AND entity_id = ?",
            (session.id, str(command.review_id)),
        ).fetchone(),
    )
    if linked is None:
        raise ValueError("review belongs to another coordinator")
    if services.review_decision is None:
        return "unavailable", "Controller credential is unavailable."
    if not command.decided_by.strip():
        raise ValueError("decision identity is required")
    review = decide_goal_review(
        graph,
        services,
        GoalReviewDecisionRequest(command.review_id, command.decision),
        command.decided_by.strip(),
    )
    return "accepted", review
