"""MCP tools for explicit top-level goal review and execution admission."""

from __future__ import annotations

from milknado.domains.graph import CommandReceipt, GoalReviewRecord, GoalReviewRequest
from milknado.mcp._core import Response, mcp, open_graph, resolve_project_root


def _receipt_response(receipt: CommandReceipt) -> dict[str, object]:
    return {
        "command_id": receipt.command_id,
        "status": receipt.status,
        "node_id": receipt.node_id,
        "run_id": receipt.run_id,
        "invocation_id": receipt.invocation_id,
        "owner_incarnation": receipt.owner_incarnation,
        "action": receipt.action,
        "expires_at": receipt.expires_at,
        "admitted_at": receipt.admitted_at,
        "recorded_at": receipt.recorded_at,
        "detail": receipt.detail,
    }


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
        "interrupt_receipts": tuple(
            _receipt_response(receipt) for receipt in record.interruption_receipts
        ),
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
