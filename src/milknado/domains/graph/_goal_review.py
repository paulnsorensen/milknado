"""Top-level goal review records and graph-owned execution admission."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import cast

from milknado.domains.common import NodeStatus
from milknado.domains.graph._goal_review_scope import (
    ancestor_chain,
    nearest_execution_goal,
    scope_ids,
    top_level_goal,
    validate_scope,
)
from milknado.domains.graph._goal_review_scope import value as _value
from milknado.domains.graph._sqlite_rows import fetchall, fetchone
from milknado.domains.graph.goal_review import (
    GoalAdmission,
    GoalAdmissionDenied,
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    GoalReviewRequest,
)

READY_NODE_ADMISSION_CTE = """
WITH RECURSIVE unbounded_review_nodes(id) AS (
    SELECT goal_id FROM goal_reviews
    WHERE decision = 'pending' AND affected_node_ids IS NULL
    UNION
    SELECT nodes.id
    FROM nodes
    JOIN unbounded_review_nodes ON nodes.parent_id = unbounded_review_nodes.id
),
paused_review_nodes(id) AS (
    SELECT id FROM unbounded_review_nodes
    UNION
    SELECT CAST(scope.value AS INTEGER)
    FROM goal_reviews
    JOIN json_each(goal_reviews.affected_node_ids) AS scope
    WHERE goal_reviews.decision = 'pending'
)
"""
READY_NODE_ADMISSION_FILTER = "n.id NOT IN (SELECT id FROM paused_review_nodes)"


def _text(value: str, label: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _record(row: object) -> GoalReviewRecord:
    raw_ids = _value(row, "affected_node_ids")
    affected = None if raw_ids is None else tuple(cast(list[int], json.loads(cast(str, raw_ids))))
    return GoalReviewRecord(
        review_id=cast(int, _value(row, "review_id")),
        goal_id=cast(int, _value(row, "goal_id")),
        goal_revision=cast(str, _value(row, "goal_revision")),
        evidence=cast(str, _value(row, "evidence")),
        proposed_change=cast(str, _value(row, "proposed_change")),
        decision=GoalReviewDecision(cast(str, _value(row, "decision"))),
        affected_node_ids=affected,
        reviewer=cast(str, _value(row, "reviewer")),
        assessed_at=cast(str, _value(row, "assessed_at")),
        decided_at=cast(str | None, _value(row, "decided_at")),
        decided_by=cast(str | None, _value(row, "decided_by")),
    )


def request_goal_review(conn: sqlite3.Connection, request: GoalReviewRequest) -> GoalReviewRecord:
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        goal_id = top_level_goal(conn, request.goal_id)
        revision = _text(request.goal_revision, "goal_revision")
        evidence = _text(request.evidence, "evidence")
        proposed_change = _text(request.proposed_change, "proposed_change")
        reviewer = _text(request.reviewer, "reviewer")
        affected = validate_scope(conn, goal_id, request.affected_node_ids)
        latest = latest_goal_review(conn, goal_id)
        if latest is not None and latest.decision is GoalReviewDecision.PENDING:
            raise ValueError(f"goal {goal_id} already has a pending review")
        assessed_at = request.assessed_at or datetime.now(UTC).isoformat()
        cursor = conn.execute(
            "INSERT INTO goal_reviews "
            + "(goal_id, goal_revision, evidence, proposed_change, decision, "
            + "affected_node_ids, reviewer, assessed_at, decided_at, decided_by) "
            + "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, NULL, NULL)",
            (
                goal_id,
                revision,
                evidence,
                proposed_change,
                None if affected is None else json.dumps(list(affected)),
                reviewer,
                assessed_at,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("goal review insert did not return an id")
        review_id = cursor.lastrowid
    row = fetchone(conn, "SELECT * FROM goal_reviews WHERE review_id = ?", (review_id,))
    if row is None:
        raise RuntimeError("goal review insert did not return a record")
    return _record(row)


def get_goal_review(conn: sqlite3.Connection, review_id: int) -> GoalReviewRecord | None:
    row = fetchone(conn, "SELECT * FROM goal_reviews WHERE review_id = ?", (review_id,))
    return _record(row) if row is not None else None


def latest_goal_review(conn: sqlite3.Connection, goal_id: int) -> GoalReviewRecord | None:
    goal_id = top_level_goal(conn, goal_id)
    row = fetchone(
        conn,
        "SELECT * FROM goal_reviews WHERE goal_id = ? ORDER BY review_id DESC LIMIT 1",
        (goal_id,),
    )
    return _record(row) if row is not None else None


def decide_goal_review(
    conn: sqlite3.Connection, request: GoalReviewDecisionRequest
) -> GoalReviewRecord:
    decision = GoalReviewDecision(request.decision)
    if decision is GoalReviewDecision.PENDING:
        raise ValueError("review decision must be accepted or rejected")
    reviewer = _text(request.reviewer, "reviewer")
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        current = get_goal_review(conn, request.review_id)
        if current is None:
            raise ValueError(f"goal review {request.review_id} not found")
        if current.decision is not GoalReviewDecision.PENDING:
            raise ValueError(
                f"goal review {request.review_id} is already {current.decision.value}"
            )
        decided_at = request.decided_at or datetime.now(UTC).isoformat()
        updated = conn.execute(
            "UPDATE goal_reviews SET decision = ?, decided_at = ?, decided_by = ? "
            + "WHERE review_id = ? AND decision = 'pending'",
            (decision.value, decided_at, reviewer, request.review_id),
        )
        if updated.rowcount != 1:
            raise ValueError(f"goal review {request.review_id} changed before decision")
    result = get_goal_review(conn, request.review_id)
    if result is None:
        raise RuntimeError("goal review disappeared after decision")
    return result


def _blocking_pending_review(
    conn: sqlite3.Connection, node_id: int, ancestor_ids: tuple[int, ...]
) -> GoalReviewRecord | None:
    """Find a pending review, on any ancestor, whose scope covers node_id.

    Mirrors READY_NODE_ADMISSION_CTE's ancestor semantics: an unbounded review
    covers every descendant of its goal, and a bounded review covers exactly
    its recorded affected_node_ids (validated to lie within the goal's scope).
    """
    if not ancestor_ids:
        return None
    placeholders = ",".join("?" for _ in ancestor_ids)
    rows = fetchall(
        conn,
        "SELECT * FROM goal_reviews WHERE decision = 'pending' AND goal_id IN ("
        + placeholders
        + ")",
        ancestor_ids,
    )
    for row in rows:
        record = _record(row)
        if record.affected_node_ids is None or node_id in record.affected_node_ids:
            return record
    return None


def goal_admission(conn: sqlite3.Connection, node_id: int) -> GoalAdmission:
    if fetchone(conn, "SELECT id FROM nodes WHERE id = ?", (node_id,)) is None:
        return GoalAdmission(False, None, None, None, None, f"node {node_id} not found")
    chain = ancestor_chain(conn, node_id)
    goal_id = nearest_execution_goal(chain, node_id)
    review = latest_goal_review(conn, goal_id) if goal_id is not None else None
    blocking = _blocking_pending_review(conn, node_id, tuple(chain))
    if blocking is not None:
        return GoalAdmission(
            False,
            blocking.goal_id,
            blocking.review_id,
            blocking.decision,
            blocking.affected_node_ids,
            f"goal {blocking.goal_id} review pending; execution paused",
        )
    return GoalAdmission(
        True,
        goal_id,
        review.review_id if review else None,
        review.decision if review else None,
        review.affected_node_ids if review else None,
    )


def assert_admitted(conn: sqlite3.Connection, node_id: int) -> None:
    admission = goal_admission(conn, node_id)
    if not admission.allowed:
        raise GoalAdmissionDenied(admission)


def interruption_targets(conn: sqlite3.Connection, review_id: int) -> tuple[int, ...]:
    review = get_goal_review(conn, review_id)
    if review is None:
        raise ValueError(f"goal review {review_id} not found")
    latest = latest_goal_review(conn, review.goal_id)
    if (
        review.decision is not GoalReviewDecision.PENDING
        or latest is None
        or latest.review_id != review_id
    ):
        return ()
    ids = (
        scope_ids(conn, review.goal_id)
        if review.unbounded
        else set(review.affected_node_ids or ())
    )
    if not ids:
        return ()
    placeholders = ",".join("?" for _ in ids)
    rows = fetchall(
        conn,
        "SELECT id FROM nodes WHERE status = ? AND id IN (" + placeholders + ") ORDER BY id",
        (NodeStatus.RUNNING.value, *sorted(ids)),
    )
    return tuple(cast(int, _value(row, "id")) for row in rows)
