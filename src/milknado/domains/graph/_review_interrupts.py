"""Durable owner-fenced interrupts emitted for pending goal reviews."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import cast

from milknado.domains.graph._command_persistence import admit_command
from milknado.domains.graph._command_records import get_capabilities, utc_iso
from milknado.domains.graph._goal_review import get_goal_review, interruption_targets
from milknado.domains.graph._sqlite_rows import fetchone
from milknado.domains.graph.commands import CommandReceipt, GraphCommand, new_command_id


def enqueue_goal_review_interrupts(
    conn: sqlite3.Connection, review_id: int, *, now: str | None = None
) -> tuple[CommandReceipt, ...]:
    review = get_goal_review(conn, review_id)
    if review is None:
        raise ValueError(f"goal review {review_id} not found")
    timestamp = utc_iso(now or review.assessed_at)
    expires_at = (datetime.fromisoformat(timestamp) + timedelta(hours=1)).isoformat()
    receipts: list[CommandReceipt] = []
    for node_id in interruption_targets(conn, review_id):
        row = fetchone(conn, "SELECT run_id FROM nodes WHERE id = ?", (node_id,))
        if row is None or (run_id := cast(str | None, row[0])) is None:
            continue
        capabilities = get_capabilities(conn, run_id)
        if capabilities is None:
            continue
        receipts.append(
            admit_command(
                conn,
                GraphCommand(
                    command_id=new_command_id(),
                    node_id=node_id,
                    run_id=run_id,
                    invocation_id=capabilities.invocation_id,
                    owner_incarnation=capabilities.owner_incarnation,
                    action="interrupt",
                    text=f"goal review {review_id} pending",
                    expires_at=expires_at,
                ),
                now=timestamp,
            )
        )
    return tuple(receipts)


__all__ = ["enqueue_goal_review_interrupts"]
