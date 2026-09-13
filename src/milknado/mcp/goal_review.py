"""MCP tools for explicit top-level goal review and execution admission."""

from __future__ import annotations

from typing import Literal

from milknado.domains.graph import (
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    GoalReviewRequest,
)
from milknado.mcp._core import Response, mcp, open_graph, resolve_project_root

ReviewDecision = Literal["accepted", "rejected"]


def _record_response(record: GoalReviewRecord) -> Response:
    return {
        "review_id": record.review_id,
        "goal_id": record.goal_id,
        "goal_revision": record.goal_revision,
        "evidence": record.evidence,
        "proposed_change": record.proposed_change,
        "decision": record.decision.value,
        "affected_node_ids": record.affected_node_ids,
        "unbounded": record.unbounded,
        "reviewer": record.reviewer,
        "assessed_at": record.assessed_at,
        "decided_at": record.decided_at,
        "decided_by": record.decided_by,
    }


@mcp.tool()
def milknado_goal_review_request(  # noqa: PLR0913 - MCP boundary schema
    goal_id: int,
    goal_revision: str,
    evidence: str,
    proposed_change: str,
    reviewer: str,
    affected_node_ids: list[int] | None = None,
    project_root: str = "",
) -> Response:
    """Pause affected execution while a possible top-level goal change is reviewed."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        record = graph.request_goal_review(
            GoalReviewRequest(
                goal_id=goal_id,
                goal_revision=goal_revision,
                evidence=evidence,
                proposed_change=proposed_change,
                affected_node_ids=(
                    tuple(affected_node_ids) if affected_node_ids is not None else None
                ),
                reviewer=reviewer,
            )
        )
        targets = graph.goal_review_interruption_targets(record.review_id)
        response = _record_response(record)
        response["links"] = {
            "goal": {"kind": "node", "node_id": record.goal_id},
            "safe_interruption_targets": tuple(
                {"kind": "node", "node_id": node_id} for node_id in targets
            ),
        }
        return response
    finally:
        graph.close()


@mcp.tool()
def milknado_goal_review_decide(
    review_id: int,
    decision: ReviewDecision,
    reviewer: str,
    project_root: str = "",
) -> Response:
    """Accept or reject one pending top-level goal change review."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        record = graph.decide_goal_review(
            GoalReviewDecisionRequest(
                review_id=review_id,
                decision=GoalReviewDecision(decision),
                reviewer=reviewer,
            )
        )
        return _record_response(record)
    finally:
        graph.close()


@mcp.tool()
def milknado_goal_admission(node_id: int, project_root: str = "") -> Response:
    """Return the graph-owned execution admission decision for one node."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        admission = graph.goal_admission(node_id)
        return {
            "allowed": admission.allowed,
            "goal_id": admission.goal_id,
            "review_id": admission.review_id,
            "decision": admission.decision.value if admission.decision else None,
            "affected_node_ids": admission.affected_node_ids,
            "reason": admission.reason,
        }
    finally:
        graph.close()
