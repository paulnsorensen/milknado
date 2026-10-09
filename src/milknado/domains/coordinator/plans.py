"""Durable planning operations for coordinator retries."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import msgspec

from milknado.domains.planning import PlanResult


class _PlanReceipt(msgspec.Struct, frozen=True):
    success: bool
    exit_code: int
    context_path: str | None
    nodes_created: int
    batch_count: int
    oversized_count: int
    solver_status: str
    change_count: int
    mega_batch_change_count: int | None


def _decode_result(payload: str) -> PlanResult:
    receipt = msgspec.json.decode(payload, type=_PlanReceipt)
    return PlanResult(
        success=receipt.success,
        exit_code=receipt.exit_code,
        context_path=Path(receipt.context_path) if receipt.context_path is not None else None,
        nodes_created=receipt.nodes_created,
        batch_count=receipt.batch_count,
        oversized_count=receipt.oversized_count,
        solver_status=receipt.solver_status,
        change_count=receipt.change_count,
        mega_batch_change_count=receipt.mega_batch_change_count,
    )


def begin_plan(
    conn: sqlite3.Connection, session_id: str, operation_id: str
) -> tuple[bool, PlanResult | None]:
    if not operation_id:
        raise ValueError("planning operation identity must not be empty")
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
    receipt = _PlanReceipt(
        result.success,
        result.exit_code,
        str(result.context_path) if result.context_path else None,
        result.nodes_created,
        result.batch_count,
        result.oversized_count,
        result.solver_status,
        result.change_count,
        result.mega_batch_change_count,
    )
    with conn:
        cursor = conn.execute(
            "UPDATE coordinator_plans SET result_json = ? "
            + "WHERE operation_id = ? AND result_json IS NULL",
            (msgspec.json.encode(receipt).decode(), operation_id),
        )
    if cursor.rowcount != 1:
        raise ValueError("planning operation result was already recorded")
