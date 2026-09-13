"""Indexed dispatch-readiness reads shared by execution and observers."""

from __future__ import annotations

import sqlite3
from typing import cast

from milknado.domains.common import NodeStatus
from milknado.domains.graph._goal_review_sql import (
    READY_NODE_ADMISSION_CTE,
    READY_NODE_ADMISSION_FILTER,
)
from milknado.domains.graph._sqlite_rows import fetchall, fetchone

_READY_CTE = (
    READY_NODE_ADMISSION_CTE
    + ", ready AS (SELECT n.id FROM nodes n WHERE "
    + READY_NODE_ADMISSION_FILTER
    + " AND n.status = ? AND n.archived_at IS NULL "
    + "AND n.id IN (SELECT child_id FROM edges) "
    + "AND NOT EXISTS (SELECT 1 FROM edges e JOIN nodes c ON c.id = e.child_id "
    + "WHERE e.parent_id = n.id AND c.status != 'done')) "
)


def ready_node_ids(conn: sqlite3.Connection) -> list[int]:
    rows = fetchall(
        conn,
        _READY_CTE + "SELECT id FROM ready ORDER BY id",
        (NodeStatus.PENDING.value,),
    )
    return [cast(int, row[0]) for row in rows]


def dispatchable_count(conn: sqlite3.Connection) -> int:
    row = fetchone(
        conn,
        _READY_CTE
        + "SELECT COUNT(*) FROM ready r WHERE NOT EXISTS ("
        + "SELECT 1 FROM file_ownership owned "
        + "JOIN file_ownership rival ON rival.file_path = owned.file_path "
        + "JOIN nodes other ON other.id = rival.node_id "
        + "WHERE owned.node_id = r.id AND rival.node_id != r.id AND ("
        + "other.status = 'running' OR (rival.node_id < r.id "
        + "AND rival.node_id IN (SELECT id FROM ready))))",
        (NodeStatus.PENDING.value,),
    )
    if row is None:
        raise RuntimeError("dispatchable node count returned no row")
    return cast(int, row[0])
