"""Durable owner-fenced interrupts emitted for pending goal reviews."""

from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timedelta
from typing import cast

import milknado.domains.graph._goal_review as _goal_review
from milknado.domains.graph._command_persistence import admit_command
from milknado.domains.graph._command_records import get_capabilities, utc_iso
from milknado.domains.graph._sqlite_rows import fetchone
from milknado.domains.graph.commands import CommandReceipt, GraphCommand, new_command_id
from milknado.domains.graph.goal_review import GoalReviewRecord, GoalReviewRequest


def request_with_interrupts(
    conn: sqlite3.Connection, request: GoalReviewRequest
) -> GoalReviewRecord:
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        record = _goal_review.request_goal_review(conn, request, _in_transaction=True)
        receipts = enqueue_goal_review_interrupts(
            conn, record.review_id, now=record.assessed_at, _in_transaction=True
        )
        return replace(record, interruption_receipts=receipts)


def enqueue_goal_review_interrupts(
    conn: sqlite3.Connection,
    review_id: int,
    *,
    now: str | None = None,
    _in_transaction: bool = False,
) -> tuple[CommandReceipt, ...]:
    if not _in_transaction:
        _ = conn.execute("BEGIN IMMEDIATE")
    with conn if not _in_transaction else nullcontext():
        review = _goal_review.get_goal_review(conn, review_id)
        if review is None:
            raise ValueError(f"goal review {review_id} not found")
        timestamp = utc_iso(now or review.assessed_at)
        expires_at = (datetime.fromisoformat(timestamp) + timedelta(hours=1)).isoformat()
        receipts: list[CommandReceipt] = []
        for node_id in _goal_review.interruption_targets(conn, review_id):
            row = fetchone(conn, "SELECT run_id FROM nodes WHERE id = ?", (node_id,))
            if row is None or (run_id := cast(str | None, row[0])) is None:
                continue
            capabilities = get_capabilities(conn, run_id)
            if capabilities is None:
                continue
            receipt = admit_command(
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
                _in_transaction=_in_transaction,
            )
            if receipt.status != "queued":
                raise ValueError(
                    receipt.detail
                    or f"interrupt command {receipt.command_id} was {receipt.status}"
                )
            receipts.append(receipt)
        return tuple(receipts)


__all__ = ["enqueue_goal_review_interrupts", "request_with_interrupts"]
