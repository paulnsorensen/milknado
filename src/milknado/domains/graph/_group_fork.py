"""Clone an execution group's tasks into an isolated graph alternative."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from milknado.domains.graph._sqlite_rows import fetchall


def clone_alternative(
    conn: sqlite3.Connection, source_group_id: str, source_graph_id: str
) -> tuple[str, tuple[int, ...]]:
    rows = fetchall(
        conn,
        "SELECT n.id, n.description, n.parent_id, n.flavor "
        + "FROM execution_group_tasks t JOIN nodes n ON n.id = t.node_id "
        + "WHERE t.group_id = ? ORDER BY t.position",
        (source_group_id,),
    )
    if not rows:
        raise ValueError("source execution group has no tasks")
    graph_id = uuid4().hex
    _ = conn.execute(
        "INSERT INTO graph_alternatives (id, source_graph_id, source_group_id, created_at) "
        + "VALUES (?, ?, ?, ?)",
        (graph_id, source_graph_id, source_group_id, datetime.now(UTC).isoformat()),
    )
    task_ids: dict[int, int] = {}
    for row in rows:
        old_id = cast(int, row[0])
        cursor = conn.execute(
            "INSERT INTO nodes (description, status, parent_id, created_at, kind, flavor) "
            + "VALUES (?, 'pending', ?, ?, 'task', ?)",
            (
                cast(str, row[1]),
                None,
                datetime.now(UTC).isoformat(),
                cast(str | None, row[3]),
            ),
        )
        task_ids[old_id] = cast(int, cursor.lastrowid)
    for row in rows:
        old_id = cast(int, row[0])
        new_id = task_ids[old_id]
        old_parent_id = cast(int | None, row[2])
        if old_parent_id in task_ids:
            _ = conn.execute(
                "UPDATE nodes SET parent_id = ? WHERE id = ?",
                (task_ids[old_parent_id], new_id),
            )
        _ = conn.execute(
            "INSERT INTO file_ownership (node_id, file_path) "
            + "SELECT ?, file_path FROM file_ownership WHERE node_id = ?",
            (new_id, old_id),
        )
        for edge in fetchall(conn, "SELECT child_id FROM edges WHERE parent_id = ?", (old_id,)):
            child_id = cast(int, edge[0])
            _ = conn.execute(
                "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
                (new_id, task_ids.get(child_id, child_id)),
            )
    return graph_id, tuple(task_ids.values())
