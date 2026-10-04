"""Durable execution-group membership, workspace identity, and writer admission."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

from milknado.domains.graph._sqlite_rows import fetchall, fetchone

CREATE_EXECUTION_GROUPS = (
    "CREATE TABLE IF NOT EXISTS execution_groups ("
    "id TEXT PRIMARY KEY, graph_id TEXT NOT NULL, "
    "worktree_path TEXT NOT NULL UNIQUE, branch_name TEXT NOT NULL UNIQUE, "
    "provider_session_id TEXT NOT NULL UNIQUE, "
    "source_group_id TEXT REFERENCES execution_groups(id), "
    "active_node_id INTEGER REFERENCES nodes(id), active_run_id TEXT)"
)
CREATE_EXECUTION_GROUP_TASKS = (
    "CREATE TABLE IF NOT EXISTS execution_group_tasks ("
    "group_id TEXT NOT NULL REFERENCES execution_groups(id) ON DELETE CASCADE, "
    "node_id INTEGER NOT NULL UNIQUE REFERENCES nodes(id), position INTEGER NOT NULL, "
    "status TEXT CHECK (status IN ('done', 'failed', 'blocked')), result TEXT, "
    "PRIMARY KEY (group_id, node_id), UNIQUE (group_id, position))"
)


@dataclass(frozen=True)
class GroupWorkspace:
    worktree_path: str
    branch_name: str
    provider_session_id: str


@dataclass(frozen=True)
class ExecutionGroup:
    id: str
    graph_id: str
    worktree_path: str
    branch_name: str
    provider_session_id: str
    source_group_id: str | None = None


@dataclass(frozen=True)
class TaskOutcome:
    status: str
    result: str


class ExecutionGroupStore:
    _db_path: Path

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        _ = conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _validate_membership(
        conn: sqlite3.Connection, graph_id: str, tasks: tuple[int, ...]
    ) -> None:
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("execution group requires distinct tasks")
        slots = ",".join("?" for _ in tasks)
        rows = fetchall(
            conn, f"SELECT id FROM nodes WHERE kind = 'task' AND id IN ({slots})", tasks
        )
        if {cast(int, row[0]) for row in rows} != set(tasks):
            raise ValueError("execution group contains an unknown or non-task node")
        order = {node_id: index for index, node_id in enumerate(tasks)}
        edges = fetchall(
            conn,
            f"SELECT parent_id, child_id FROM edges WHERE parent_id IN ({slots}) "
            + f"AND child_id IN ({slots})",
            (*tasks, *tasks),
        )
        if any(order[cast(int, row[1])] >= order[cast(int, row[0])] for row in edges):
            raise ValueError("execution group violates dependency order")
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

    def create(
        self, graph_id: str, tasks: tuple[int, ...], workspace: GroupWorkspace
    ) -> ExecutionGroup:
        return self._create(graph_id, tasks, workspace, None)

    def _create(
        self,
        graph_id: str,
        tasks: tuple[int, ...],
        workspace: GroupWorkspace,
        source_group_id: str | None,
    ) -> ExecutionGroup:
        identities = (
            graph_id,
            workspace.worktree_path,
            workspace.branch_name,
            workspace.provider_session_id,
        )
        if not all(identities):
            raise ValueError("execution group identities must be nonempty")
        group = ExecutionGroup(uuid4().hex, graph_id, *identities[1:], source_group_id)
        with closing(self._connect()) as conn, conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            self._validate_membership(conn, graph_id, tasks)
            try:
                _ = conn.execute(
                    "INSERT INTO execution_groups "
                    + "(id, graph_id, worktree_path, branch_name, "
                    + "provider_session_id, source_group_id) "
                    + "VALUES (?, ?, ?, ?, ?, ?)",
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
                    ((group.id, node_id, index) for index, node_id in enumerate(tasks)),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("execution group identity or task already exists") from error
        return group

    def fork(
        self, source_group_id: str, tasks: tuple[int, ...], workspace: GroupWorkspace
    ) -> ExecutionGroup:
        source = self.get(source_group_id)
        if source is None:
            raise ValueError("source execution group does not exist")
        if (
            workspace.worktree_path == source.worktree_path
            or workspace.branch_name == source.branch_name
            or workspace.provider_session_id == source.provider_session_id
        ):
            raise ValueError("fork must have distinct workspace identities")
        return self._create(uuid4().hex, tasks, workspace, source_group_id)

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

    def start_task(self, group_id: str, node_id: int, run_id: str) -> None:  # noqa: V105
        if not run_id:
            raise ValueError("run identity must be nonempty")
        with closing(self._connect()) as conn, conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            group = fetchone(
                conn, "SELECT active_run_id FROM execution_groups WHERE id = ?", (group_id,)
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
            predecessor = fetchone(
                conn,
                "SELECT 1 FROM execution_group_tasks WHERE group_id = ? "
                + "AND position < ? AND status IS NOT 'done' LIMIT 1",
                (group_id, cast(int, member[0])),
            )
            if predecessor is not None:
                raise ValueError("execution group predecessor has not completed")
            _ = conn.execute(
                "UPDATE execution_groups SET active_node_id = ?, active_run_id = ? WHERE id = ?",
                (node_id, run_id, group_id),
            )

    def finish_task(self, group_id: str, run_id: str, outcome: TaskOutcome) -> None:  # noqa: V105
        if outcome.status not in {"done", "failed", "blocked"}:
            raise ValueError("invalid task result status")
        with closing(self._connect()) as conn, conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            writer = fetchone(
                conn,
                "SELECT active_node_id FROM execution_groups WHERE id = ? AND active_run_id = ?",
                (group_id, run_id),
            )
            if writer is None:
                raise ValueError("execution group writer fence lost")
            _ = conn.execute(
                "UPDATE execution_group_tasks SET status = ?, result = ? "
                + "WHERE group_id = ? AND node_id = ?",
                (outcome.status, outcome.result, group_id, cast(int, writer[0])),
            )
            _ = conn.execute(
                "UPDATE execution_groups SET active_node_id = NULL, active_run_id = NULL "
                + "WHERE id = ?",
                (group_id,),
            )

    def task_result(self, node_id: int) -> tuple[str, str] | None:  # noqa: V105
        with closing(self._connect()) as conn:
            row = fetchone(
                conn,
                "SELECT status, result FROM execution_group_tasks WHERE node_id = ?",
                (node_id,),
            )
        return (cast(str, row[0]), cast(str, row[1])) if row and row[0] is not None else None
