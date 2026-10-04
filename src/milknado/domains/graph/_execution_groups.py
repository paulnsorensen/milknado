"""Durable execution-group identity, membership, and writer admission."""

from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager, closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from milknado.domains.common import NodeStatus
from milknado.domains.graph._group_fork import clone_alternative
from milknado.domains.graph._group_models import (
    ExecutionGroup,
    GroupPlan,
    GroupWorkspace,
    TaskAttempt,
    TaskOutcome,
)
from milknado.domains.graph._group_policy import validate_membership
from milknado.domains.graph._group_reservation import admit_writer, reserved_workspace
from milknado.domains.graph._sqlite_rows import fetchall, fetchone
from milknado.domains.graph.goal_review import GoalAdmission

__all__ = ["ExecutionGroup", "ExecutionGroupStore", "GroupWorkspace", "TaskAttempt", "TaskOutcome"]


class _GroupGraph(Protocol):
    @property
    def db_path(self) -> Path: ...

    @property
    def synchronization_lock(self) -> AbstractContextManager[object]: ...

    @property
    def group_connection(self) -> sqlite3.Connection: ...

    def group_notifications(self) -> AbstractContextManager[None]: ...

    def claim_group_node(self, node_id: int, run_id: str, *, now: str) -> bool: ...
    def goal_admission(self, node_id: int) -> GoalAdmission: ...
    def mark_terminal(self, node_id: int, run_id: str, status: NodeStatus) -> bool: ...
    def mark_failed(self, node_id: int) -> None: ...
    def mark_blocked_fenced(self, node_id: int, run_id: str) -> bool: ...


class ExecutionGroupStore:
    _graph: _GroupGraph

    def __init__(self, graph: _GroupGraph) -> None:
        self._graph = graph

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._graph.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        _ = conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _persist(conn: sqlite3.Connection, plan: GroupPlan) -> ExecutionGroup:
        workspace = plan.workspace
        if not all(
            (
                plan.graph_id,
                workspace.worktree_path,
                workspace.branch_name,
                workspace.provider_session_id,
            )
        ):
            raise ValueError("execution group identities must be nonempty")
        validate_membership(conn, plan.graph_id, plan.tasks)
        group = ExecutionGroup(
            uuid4().hex,
            plan.graph_id,
            workspace.worktree_path,
            workspace.branch_name,
            workspace.provider_session_id,
            plan.source_group_id,
        )
        try:
            _ = conn.execute(
                "INSERT OR IGNORE INTO graph_alternatives "
                + "(id, source_graph_id, source_group_id, created_at) VALUES (?, NULL, NULL, ?)",
                (plan.graph_id, datetime.now(UTC).isoformat()),
            )
            _ = conn.execute(
                "INSERT INTO execution_groups "
                + "(id, graph_id, worktree_path, branch_name, provider_session_id, "
                + "source_group_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    group.id,
                    group.graph_id,
                    group.worktree_path,
                    group.branch_name,
                    group.provider_session_id,
                    group.source_group_id,
                ),
            )
            _ = conn.executemany(
                "INSERT INTO execution_group_tasks (group_id, node_id, position) "
                + "VALUES (?, ?, ?)",
                ((group.id, node_id, index) for index, node_id in enumerate(plan.tasks)),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("execution group identity or task already exists") from error
        return group

    def create(
        self, graph_id: str, tasks: tuple[int, ...], workspace: GroupWorkspace
    ) -> ExecutionGroup:
        with self._graph.synchronization_lock, closing(self._connect()) as conn, conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            return self._persist(conn, GroupPlan(graph_id, tasks, workspace))

    def fork(self, source_group_id: str, workspace: GroupWorkspace) -> ExecutionGroup:
        with self._graph.synchronization_lock, closing(self._connect()) as conn, conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            source = self.get(source_group_id)
            if source is None:
                raise ValueError("source execution group does not exist")
            if (
                workspace.worktree_path == source.worktree_path
                or workspace.branch_name == source.branch_name
                or workspace.provider_session_id == source.provider_session_id
            ):
                raise ValueError("fork must have distinct workspace identities")
            graph_id, tasks = clone_alternative(conn, source.id, source.graph_id)
            return self._persist(conn, GroupPlan(graph_id, tasks, workspace, source.id))

    def get(self, group_id: str) -> ExecutionGroup | None:
        with closing(self._connect()) as conn:
            row = fetchone(
                conn,
                "SELECT id, graph_id, worktree_path, branch_name, provider_session_id, "
                + "source_group_id FROM execution_groups WHERE id = ?",
                (group_id,),
            )
        if row is None:
            return None
        return ExecutionGroup(
            cast(str, row[0]),
            cast(str, row[1]),
            cast(str, row[2]),
            cast(str, row[3]),
            cast(str, row[4]),
            cast(str | None, row[5]),
        )

    def tasks(self, group_id: str) -> tuple[int, ...]:
        with closing(self._connect()) as conn:
            rows = fetchall(
                conn,
                "SELECT node_id FROM execution_group_tasks WHERE group_id = ? ORDER BY position",
                (group_id,),
            )
        return tuple(cast(int, row[0]) for row in rows)

    def for_task(self, node_id: int) -> ExecutionGroup | None:
        with closing(self._connect()) as conn:
            row = fetchone(
                conn,
                "SELECT group_id FROM execution_group_tasks WHERE node_id = ?",
                (node_id,),
            )
        return self.get(cast(str, row[0])) if row is not None else None

    def active_attempt(self, group_id: str) -> TaskAttempt | None:
        with closing(self._connect()) as conn:
            row = fetchone(
                conn,
                "SELECT active_node_id, active_run_id, active_attempt_id "
                + "FROM execution_groups WHERE id = ?",
                (group_id,),
            )
        if row is None or row[2] is None:
            return None
        return TaskAttempt(group_id, cast(int, row[0]), cast(str, row[1]), cast(str, row[2]))

    def reserve_task(self, group_id: str, node_id: int, run_id: str) -> TaskAttempt:
        if not run_id:
            raise ValueError("run identity must be nonempty")
        with self._graph.synchronization_lock, closing(self._connect()) as conn, conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            attempt, _ = admit_writer(conn, group_id, node_id, run_id)
            return attempt

    def launch_reserved_task(self, attempt: TaskAttempt) -> None:
        with self._graph.synchronization_lock, self._graph.group_notifications():
            conn = self._graph.group_connection
            with conn:
                _ = conn.execute("BEGIN IMMEDIATE")
                workspace = reserved_workspace(conn, attempt)
                if not self._graph.goal_admission(attempt.node_id).allowed:
                    raise ValueError("goal review pauses task launch")
                row = fetchone(
                    conn, "SELECT status, run_id FROM nodes WHERE id = ?", (attempt.node_id,)
                )
                if row is not None and (row[0], row[1]) == ("running", attempt.attempt_id):
                    return
                if not self._graph.claim_group_node(
                    attempt.node_id, attempt.attempt_id, now=datetime.now(UTC).isoformat()
                ):
                    raise ValueError("execution group task is not ready")
                cursor = conn.execute(
                    "UPDATE nodes SET worktree_path = ?, branch_name = ? "
                    + "WHERE id = ? AND run_id = ? AND status = 'running'",
                    (
                        workspace.worktree_path,
                        workspace.branch_name,
                        attempt.node_id,
                        attempt.attempt_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("execution group writer fence lost")

    def fail_reserved_task(self, attempt: TaskAttempt, reason: str) -> None:
        if not reason:
            raise ValueError("launch failure reason must be nonempty")
        with self._graph.synchronization_lock, self._graph.group_notifications():
            conn = self._graph.group_connection
            with conn:
                _ = conn.execute("BEGIN IMMEDIATE")
                _ = reserved_workspace(conn, attempt)
                node = fetchone(
                    conn, "SELECT status, run_id FROM nodes WHERE id = ?", (attempt.node_id,)
                )
                if node is None or (node[0], node[1]) != ("pending", None):
                    raise ValueError("reserved task changed owner before launch failure")
                self._graph.mark_failed(attempt.node_id)
                _ = conn.execute(
                    "UPDATE execution_group_tasks SET status = 'failed', result = ? "
                    + "WHERE group_id = ? AND node_id = ?",
                    (reason, attempt.group_id, attempt.node_id),
                )
                _ = conn.execute(
                    "UPDATE execution_groups SET active_node_id = NULL, active_run_id = NULL, "
                    + "active_attempt_id = NULL WHERE id = ? AND active_attempt_id = ?",
                    (attempt.group_id, attempt.attempt_id),
                )

    def start_task(self, group_id: str, node_id: int, run_id: str) -> TaskAttempt:  # noqa: V105
        if not run_id:
            raise ValueError("run identity must be nonempty")
        with self._graph.synchronization_lock, self._graph.group_notifications():
            conn = self._graph.group_connection
            with conn:
                _ = conn.execute("BEGIN IMMEDIATE")
                attempt, workspace = admit_writer(conn, group_id, node_id, run_id)
                if not self._graph.claim_group_node(
                    node_id, attempt.attempt_id, now=datetime.now(UTC).isoformat()
                ):
                    raise ValueError("execution group task is not ready")
                cursor = conn.execute(
                    "UPDATE nodes SET worktree_path = ?, branch_name = ? "
                    + "WHERE id = ? AND run_id = ? AND status = 'running'",
                    (workspace.worktree_path, workspace.branch_name, node_id, attempt.attempt_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("execution group writer fence lost")
                return attempt

    def finish_task(self, attempt: TaskAttempt, outcome: TaskOutcome) -> None:  # noqa: V105
        if outcome.status not in {"done", "failed", "blocked"}:
            raise ValueError("invalid task result status")
        with self._graph.synchronization_lock, self._graph.group_notifications():
            conn = self._graph.group_connection
            with conn:
                _ = conn.execute("BEGIN IMMEDIATE")
                _ = reserved_workspace(conn, attempt)
                if outcome.status == "blocked":
                    landed = self._graph.mark_blocked_fenced(attempt.node_id, attempt.attempt_id)
                else:
                    status = NodeStatus.DONE if outcome.status == "done" else NodeStatus.FAILED
                    landed = self._graph.mark_terminal(attempt.node_id, attempt.attempt_id, status)
                if not landed:
                    raise ValueError("execution group writer fence lost")
                _ = conn.execute(
                    "UPDATE execution_group_tasks SET status = ?, result = ? "
                    + "WHERE group_id = ? AND node_id = ?",
                    (outcome.status, outcome.result, attempt.group_id, attempt.node_id),
                )
                _ = conn.execute(
                    "UPDATE execution_groups SET active_node_id = NULL, active_run_id = NULL, "
                    + "active_attempt_id = NULL WHERE id = ? AND active_attempt_id = ?",
                    (attempt.group_id, attempt.attempt_id),
                )

    def task_result(self, node_id: int) -> tuple[str, str] | None:  # noqa: V105
        with closing(self._connect()) as conn:
            row = fetchone(
                conn,
                "SELECT status, result FROM execution_group_tasks WHERE node_id = ?",
                (node_id,),
            )
        return (cast(str, row[0]), cast(str, row[1])) if row and row[0] is not None else None
