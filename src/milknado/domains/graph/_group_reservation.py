from __future__ import annotations

import sqlite3
from typing import cast
from uuid import uuid4

from milknado.domains.graph._group_models import GroupWorkspace, TaskAttempt
from milknado.domains.graph._group_policy import validate_task_prerequisites
from milknado.domains.graph._sqlite_rows import fetchone


def admit_writer(
    conn: sqlite3.Connection, group_id: str, node_id: int, run_id: str
) -> tuple[TaskAttempt, GroupWorkspace]:
    group = fetchone(
        conn,
        "SELECT active_run_id, worktree_path, branch_name, provider_session_id "
        + "FROM execution_groups WHERE id = ?",
        (group_id,),
    )
    if group is None:
        raise ValueError("execution group does not exist")
    if group[0] is not None:
        raise ValueError("execution group already has an active writer")
    member = fetchone(
        conn,
        "SELECT position, status FROM execution_group_tasks "
        + "WHERE group_id = ? AND node_id = ?",
        (group_id, node_id),
    )
    if member is None:
        raise ValueError("task is not an execution group member")
    if member[1] is not None:
        raise ValueError("execution group task already completed")
    validate_task_prerequisites(conn, group_id, node_id)
    predecessor = fetchone(
        conn,
        "SELECT 1 FROM execution_group_tasks WHERE group_id = ? "
        + "AND position < ? AND status IS NOT 'done' LIMIT 1",
        (group_id, cast(int, member[0])),
    )
    if predecessor is not None:
        raise ValueError("execution group predecessor has not completed")
    node = fetchone(
        conn,
        "SELECT n.status, n.run_id FROM nodes AS n WHERE n.id = ? "
        + "AND n.status IN ('pending', 'failed', 'blocked') "
        + "AND NOT EXISTS (SELECT 1 FROM run_workers AS w "
        + "WHERE w.node_id = n.id AND w.ended_at IS NULL)",
        (node_id,),
    )
    if node is None:
        raise ValueError("execution group task is not claimable")
    attempt = TaskAttempt(group_id, node_id, run_id, uuid4().hex)
    _ = conn.execute(
        "UPDATE execution_groups SET active_node_id = ?, active_run_id = ?, "
        + "active_attempt_id = ?, active_node_status = ?, active_node_run_id = ? WHERE id = ?",
        (node_id, run_id, attempt.attempt_id, node[0], node[1], group_id),
    )
    workspace = GroupWorkspace(cast(str, group[1]), cast(str, group[2]), cast(str, group[3]))
    return attempt, workspace


def reserved_workspace(conn: sqlite3.Connection, attempt: TaskAttempt) -> GroupWorkspace:
    row = fetchone(
        conn,
        "SELECT active_node_id, active_run_id, worktree_path, branch_name, provider_session_id "
        + "FROM execution_groups WHERE id = ? AND active_attempt_id = ?",
        (attempt.group_id, attempt.attempt_id),
    )
    if row is None or (row[0], row[1]) != (attempt.node_id, attempt.run_id):
        raise ValueError("execution group writer fence lost")
    return GroupWorkspace(cast(str, row[2]), cast(str, row[3]), cast(str, row[4]))


def claim_reservation_allows(
    conn: sqlite3.Connection, node_id: int, run_id: str, group_reservation: bool
) -> bool:
    row = fetchone(
        conn,
        "SELECT active_attempt_id FROM execution_groups WHERE active_node_id = ?",
        (node_id,),
    )
    if row is None:
        return not group_reservation
    return group_reservation and row[0] == run_id


def fail_reservation(conn: sqlite3.Connection, attempt: TaskAttempt, reason: str) -> None:
    reservation = fetchone(
        conn,
        "SELECT active_node_status, active_node_run_id FROM execution_groups "
        + "WHERE id = ? AND active_node_id = ? AND active_run_id = ? AND active_attempt_id = ?",
        (attempt.group_id, attempt.node_id, attempt.run_id, attempt.attempt_id),
    )
    if reservation is None or reservation[0] is None:
        raise ValueError("execution group writer fence lost")
    node = fetchone(conn, "SELECT status, run_id FROM nodes WHERE id = ?", (attempt.node_id,))
    if node is None or (node[0], node[1]) != (reservation[0], reservation[1]):
        raise ValueError("reserved task changed owner before launch failure")
    worker = fetchone(
        conn,
        "SELECT 1 FROM run_workers WHERE node_id = ? AND ended_at IS NULL LIMIT 1",
        (attempt.node_id,),
    )
    if worker is not None:
        raise ValueError("reserved task has an unresolved worker")
    _ = conn.execute(
        "UPDATE execution_group_tasks SET status = 'failed', result = ? "
        + "WHERE group_id = ? AND node_id = ?",
        (reason, attempt.group_id, attempt.node_id),
    )
    _ = conn.execute(
        "UPDATE execution_groups SET active_node_id = NULL, active_run_id = NULL, "
        + "active_attempt_id = NULL, active_node_status = NULL, active_node_run_id = NULL "
        + "WHERE id = ? AND active_attempt_id = ?",
        (attempt.group_id, attempt.attempt_id),
    )
