"""Execution-group membership and dependency policy."""

from __future__ import annotations

import sqlite3
from typing import cast

from milknado.domains.graph._sqlite_rows import fetchall, fetchone


def validate_external_prerequisites(conn: sqlite3.Connection, tasks: tuple[int, ...]) -> None:
    slots = ",".join("?" for _ in tasks)
    blocked = fetchone(
        conn,
        "SELECT 1 FROM edges e JOIN nodes child ON child.id = e.child_id "
        + f"WHERE e.parent_id IN ({slots}) AND e.child_id NOT IN ({slots}) "
        + "AND child.status != 'done' LIMIT 1",
        (*tasks, *tasks),
    )
    if blocked is not None:
        raise ValueError("execution group has an incomplete external prerequisite")


def validate_task_prerequisites(conn: sqlite3.Connection, group_id: str, node_id: int) -> None:
    blocked = fetchone(
        conn,
        "SELECT 1 FROM edges e JOIN nodes child ON child.id = e.child_id "
        + "WHERE e.parent_id = ? AND e.child_id NOT IN "
        + "(SELECT node_id FROM execution_group_tasks WHERE group_id = ?) "
        + "AND child.status != 'done' LIMIT 1",
        (node_id, group_id),
    )
    if blocked is not None:
        raise ValueError("execution group task has an incomplete external prerequisite")


def validate_membership(conn: sqlite3.Connection, graph_id: str, tasks: tuple[int, ...]) -> None:
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("execution group requires distinct tasks")
    slots = ",".join("?" for _ in tasks)
    rows = fetchall(
        conn,
        f"SELECT id FROM nodes WHERE kind = 'task' AND id IN ({slots}) "
        + "AND status IN ('pending', 'failed', 'blocked')",
        tasks,
    )
    if {cast(int, row[0]) for row in rows} != set(tasks):
        raise ValueError("execution group contains an unknown, non-task, or active node")
    order = {node_id: index for index, node_id in enumerate(tasks)}
    edges = fetchall(
        conn,
        f"SELECT parent_id, child_id FROM edges WHERE parent_id IN ({slots}) "
        + f"AND child_id IN ({slots})",
        (*tasks, *tasks),
    )
    if any(order[cast(int, row[1])] >= order[cast(int, row[0])] for row in edges):
        raise ValueError("execution group violates dependency order")
    validate_external_prerequisites(conn, tasks)
    conflict = fetchone(
        conn,
        "SELECT 1 FROM file_ownership proposed "
        + "JOIN file_ownership assigned ON proposed.file_path = assigned.file_path "
        + "JOIN execution_group_tasks members ON members.node_id = assigned.node_id "
        + "JOIN execution_groups groups ON groups.id = members.group_id "
        + f"WHERE proposed.node_id IN ({slots}) AND groups.graph_id = ? LIMIT 1",
        (*tasks, graph_id),
    )
    if conflict is not None:
        raise ValueError("execution group conflicts with file ownership")
