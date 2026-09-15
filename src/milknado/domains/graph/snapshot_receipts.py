"""Read-only durable command receipt projections for node details."""

from __future__ import annotations

import sqlite3
from typing import cast

from milknado.domains.graph._command_records import _RECEIPT_SELECT, receipt
from milknado.domains.graph.commands import CommandReceipt
from milknado.domains.graph.snapshot_history import page, table_exists
from milknado.domains.graph.snapshot_models import SnapshotPage


def receipts(
    conn: sqlite3.Connection, node_id: int, page_number: int, limit: int
) -> SnapshotPage[CommandReceipt]:
    stored = table_exists(conn, "command_receipts") and table_exists(conn, "session_commands")
    result = page(
        conn,
        _RECEIPT_SELECT + "WHERE c.node_id = ? ORDER BY r.receipt_seq LIMIT ? OFFSET ?",
        "SELECT COUNT(*) FROM command_receipts r "
        + "JOIN session_commands c ON c.command_id = r.command_id WHERE c.node_id = ?",
        node_id,
        page_number,
        limit,
        lambda row: receipt(row),
        stored,
    )
    return cast(SnapshotPage[CommandReceipt], result)
