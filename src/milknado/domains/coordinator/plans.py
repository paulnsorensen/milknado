"""Durable planning operations for coordinator retries."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import cast

from milknado.domains.planning import PlanResult


def _create_table(conn: sqlite3.Connection) -> None:
    with conn:
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS coordinator_plans (
                operation_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES coordinator_sessions(id) ON DELETE CASCADE,
                result_json TEXT
            )
        """)


def _decode_result(payload: str) -> PlanResult:
    values = cast(dict[str, object], json.loads(payload))
    context = values["context_path"]
    return PlanResult(
        success=cast(bool, values["success"]),
        exit_code=cast(int, values["exit_code"]),
        context_path=Path(cast(str, context)) if context is not None else None,
        nodes_created=cast(int, values["nodes_created"]),
        batch_count=cast(int, values["batch_count"]),
        oversized_count=cast(int, values["oversized_count"]),
        solver_status=cast(str, values["solver_status"]),
        change_count=cast(int, values["change_count"]),
        mega_batch_change_count=cast(int | None, values["mega_batch_change_count"]),
    )


def begin_plan(
    conn: sqlite3.Connection, session_id: str, operation_id: str
) -> tuple[bool, PlanResult | None]:
    if not operation_id:
        raise ValueError("planning operation identity must not be empty")
    _create_table(conn)
    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO coordinator_plans (operation_id, session_id) VALUES (?, ?)",
            (operation_id, session_id),
        )
    if cursor.rowcount == 1:
        return True, None
    row = cast(
        tuple[str, str | None] | None,
        conn.execute(
            "SELECT session_id, result_json FROM coordinator_plans WHERE operation_id = ?",
            (operation_id,),
        ).fetchone(),
    )
    if row is None or row[0] != session_id:
        raise ValueError("planning operation belongs to another coordinator")
    if row[1] is None:
        raise ValueError("planning operation has no durable result")
    return False, _decode_result(row[1])


def finish_plan(conn: sqlite3.Connection, operation_id: str, result: PlanResult) -> None:
    values = asdict(result)
    values["context_path"] = str(result.context_path) if result.context_path else None
    with conn:
        cursor = conn.execute(
            "UPDATE coordinator_plans SET result_json = ? "
            + "WHERE operation_id = ? AND result_json IS NULL",
            (json.dumps(values), operation_id),
        )
    if cursor.rowcount != 1:
        raise ValueError("planning operation result was already recorded")
