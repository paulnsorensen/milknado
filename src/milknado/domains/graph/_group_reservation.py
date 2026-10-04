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
    attempt = TaskAttempt(group_id, node_id, run_id, uuid4().hex)
    _ = conn.execute(
        "UPDATE execution_groups SET active_node_id = ?, active_run_id = ?, "
        + "active_attempt_id = ? WHERE id = ?",
        (node_id, run_id, attempt.attempt_id, group_id),
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
