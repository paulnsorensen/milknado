"""Top-level goal review records and graph-owned execution admission."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

from milknado.domains.common import NodeKind, NodeStatus
from milknado.domains.graph._sqlite_rows import fetchall, fetchone
from milknado.domains.graph.goal_review import (
    GoalAdmission,
    GoalAdmissionDenied,
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    GoalReviewRequest,
    GoalReviewSubjectError,
)


def _value(row: object, name: str) -> object:
    if isinstance(row, sqlite3.Row):
        return cast(object, row[name])
    return cast(Mapping[str, object], row)[name]


def _text(value: str, label: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _nodes(conn: sqlite3.Connection) -> dict[int, tuple[int | None, str]]:
    return {
        cast(int, _value(row, "id")): (
            cast(int | None, _value(row, "parent_id")),
            cast(str, _value(row, "kind")),
        )
        for row in fetchall(conn, "SELECT id, parent_id, kind FROM nodes")
    }


def _is_execution_goal(nodes: Mapping[int, tuple[int | None, str]], node_id: int) -> bool:
    parent_id, kind = nodes[node_id]
    if kind != NodeKind.GOAL.value or parent_id is None:
        return kind == NodeKind.GOAL.value
    parent_parent_id, parent_kind = nodes.get(parent_id, (None, ""))
    return parent_kind == NodeKind.ROADMAP.value or (
        parent_kind == NodeKind.GOAL.value and parent_parent_id is None
    )


def _top_level_goal(conn: sqlite3.Connection, node_id: int) -> int:
    nodes = _nodes(conn)
    if node_id not in nodes:
        raise GoalReviewSubjectError(f"review subject {node_id} not found")
    if _is_execution_goal(nodes, node_id):
        return node_id
    kind = nodes[node_id][1]
    subject = "nested GOAL" if kind == NodeKind.GOAL.value else kind.upper()
    raise GoalReviewSubjectError(
        f"review subject {node_id} must be an explicit execution GOAL, not {subject}"
    )


def _execution_goal(conn: sqlite3.Connection, node_id: int) -> int | None:
    nodes = _nodes(conn)
    current, seen = node_id, set[int]()
    while current not in seen and current in nodes:
        seen.add(current)
        if _is_execution_goal(nodes, current):
            return current
        parent_id = nodes[current][0]
        if parent_id is None:
            return None
        current = parent_id
    return None


def _scope_ids(conn: sqlite3.Connection, goal_id: int) -> set[int]:
    rows = fetchall(conn, "SELECT id, parent_id FROM nodes")
    children: dict[int, list[int]] = {}
    for row in rows:
        parent_id = cast(int | None, _value(row, "parent_id"))
        if parent_id is not None:
            children.setdefault(parent_id, []).append(cast(int, _value(row, "id")))
    scope, stack = set[int](), [goal_id]
    while stack:
        node_id = stack.pop()
        if node_id in scope:
            continue
        scope.add(node_id)
        stack.extend(children.get(node_id, ()))
    return scope


def _expanded_scope_ids(conn: sqlite3.Connection, roots: tuple[int, ...]) -> set[int]:
    scope: set[int] = set()
    for root_id in roots:
        scope.update(_scope_ids(conn, root_id))
    return scope


def _validate_scope(
    conn: sqlite3.Connection, goal_id: int, affected_node_ids: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    if affected_node_ids is None:
        return None
    if any(type(node_id) is not int for node_id in affected_node_ids):
        raise ValueError("affected_node_ids must contain integers")
    if len(set(affected_node_ids)) != len(affected_node_ids):
        raise ValueError("affected_node_ids contains duplicates")
    outside = sorted(set(affected_node_ids) - _scope_ids(conn, goal_id))
    if outside:
        raise ValueError(f"affected nodes are outside goal {goal_id}: {outside}")
    return tuple(affected_node_ids)


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
        goal_id = _top_level_goal(conn, request.goal_id)
        revision = _text(request.goal_revision, "goal_revision")
        evidence = _text(request.evidence, "evidence")
        proposed_change = _text(request.proposed_change, "proposed_change")
        reviewer = _text(request.reviewer, "reviewer")
        affected = _validate_scope(conn, goal_id, request.affected_node_ids)
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
    goal_id = _top_level_goal(conn, goal_id)
    row = fetchone(
        conn,
        "SELECT * FROM goal_reviews WHERE goal_id = ? ORDER BY review_id DESC LIMIT 1",
        (goal_id,),
    )
    return _record(row) if row is not None else None


def decide_goal_review(
    conn: sqlite3.Connection, request: GoalReviewDecisionRequest, *, decided_by: str
) -> GoalReviewRecord:
    decision = GoalReviewDecision(request.decision)
    if decision is GoalReviewDecision.PENDING:
        raise ValueError("review decision must be accepted or rejected")
    identity = _text(decided_by, "decided_by")
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
            (decision.value, decided_at, identity, request.review_id),
        )
        if updated.rowcount != 1:
            raise ValueError(f"goal review {request.review_id} changed before decision")
    result = get_goal_review(conn, request.review_id)
    if result is None:
        raise RuntimeError("goal review disappeared after decision")

    return result


def goal_admission(conn: sqlite3.Connection, node_id: int) -> GoalAdmission:
    if fetchone(conn, "SELECT id FROM nodes WHERE id = ?", (node_id,)) is None:
        return GoalAdmission(False, None, None, None, None, f"node {node_id} not found")
    goal_id = _execution_goal(conn, node_id)
    if goal_id is None:
        return GoalAdmission(True, None, None, None, None)
    review = latest_goal_review(conn, goal_id)
    if review is None or review.decision is not GoalReviewDecision.PENDING:
        return GoalAdmission(
            True,
            goal_id,
            review.review_id if review else None,
            review.decision if review else None,
            review.affected_node_ids if review else None,
        )
    blocked = review.affected_node_ids is None or node_id in _expanded_scope_ids(
        conn, review.affected_node_ids
    )
    if not blocked:
        return GoalAdmission(
            True,
            goal_id,
            review.review_id,
            review.decision,
            review.affected_node_ids,
        )
    return GoalAdmission(
        False,
        goal_id,
        review.review_id,
        review.decision,
        review.affected_node_ids,
        f"goal {goal_id} review pending; execution paused",
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
        _expanded_scope_ids(conn, (review.goal_id,))
        if review.unbounded
        else _expanded_scope_ids(conn, review.affected_node_ids or ())
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
