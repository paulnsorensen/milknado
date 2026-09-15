"""Ancestor-chain and scope helpers for top-level goal review admission."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import cast

from milknado.domains.common import NodeKind
from milknado.domains.graph._sqlite_rows import fetchall
from milknado.domains.graph.goal_review import GoalReviewSubjectError

NodeChain = Mapping[int, tuple[int | None, str]]


def value(row: object, name: str) -> object:
    if isinstance(row, sqlite3.Row):
        return cast(object, row[name])
    return cast(Mapping[str, object], row)[name]


def _nodes(conn: sqlite3.Connection) -> dict[int, tuple[int | None, str]]:
    return {
        cast(int, value(row, "id")): (
            cast(int | None, value(row, "parent_id")),
            cast(str, value(row, "kind")),
        )
        for row in fetchall(conn, "SELECT id, parent_id, kind FROM nodes")
    }


def _is_execution_goal(nodes: NodeChain, node_id: int) -> bool:
    parent_id, kind = nodes[node_id]
    if kind != NodeKind.GOAL.value or parent_id is None:
        return kind == NodeKind.GOAL.value
    parent_parent_id, parent_kind = nodes.get(parent_id, (None, ""))
    return parent_kind == NodeKind.ROADMAP.value or (
        parent_kind == NodeKind.GOAL.value and parent_parent_id is None
    )


def top_level_goal(conn: sqlite3.Connection, node_id: int) -> int:
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


def ancestor_chain(conn: sqlite3.Connection, node_id: int) -> dict[int, tuple[int | None, str]]:
    rows = fetchall(
        conn,
        "WITH RECURSIVE ancestors(id, parent_id, kind) AS ("
        + "SELECT id, parent_id, kind FROM nodes WHERE id = ? "
        + "UNION ALL "
        + "SELECT nodes.id, nodes.parent_id, nodes.kind FROM nodes "
        + "JOIN ancestors ON nodes.id = ancestors.parent_id"
        + ") SELECT id, parent_id, kind FROM ancestors",
        (node_id,),
    )
    return {
        cast(int, value(row, "id")): (
            cast(int | None, value(row, "parent_id")),
            cast(str, value(row, "kind")),
        )
        for row in rows
    }


def nearest_execution_goal(chain: NodeChain, node_id: int) -> int | None:
    current, seen = node_id, set[int]()
    while current not in seen and current in chain:
        seen.add(current)
        if _is_execution_goal(chain, current):
            return current
        parent_id = chain[current][0]
        if parent_id is None:
            return None
        current = parent_id
    return None


def scope_ids(conn: sqlite3.Connection, goal_id: int) -> set[int]:
    rows = fetchall(conn, "SELECT id, parent_id FROM nodes")
    children: dict[int, list[int]] = {}
    for row in rows:
        parent_id = cast(int | None, value(row, "parent_id"))
        if parent_id is not None:
            children.setdefault(parent_id, []).append(cast(int, value(row, "id")))
    scope, stack = set[int](), [goal_id]
    while stack:
        node_id = stack.pop()
        if node_id in scope:
            continue
        scope.add(node_id)
        stack.extend(children.get(node_id, ()))
    return scope


def validate_scope(
    conn: sqlite3.Connection, goal_id: int, affected_node_ids: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    if affected_node_ids is None:
        return None
    if any(type(node_id) is not int for node_id in affected_node_ids):
        raise ValueError("affected_node_ids must contain integers")
    if len(set(affected_node_ids)) != len(affected_node_ids):
        raise ValueError("affected_node_ids contains duplicates")
    outside = sorted(set(affected_node_ids) - scope_ids(conn, goal_id))
    if outside:
        raise ValueError(f"affected nodes are outside goal {goal_id}: {outside}")
    return tuple(affected_node_ids)
