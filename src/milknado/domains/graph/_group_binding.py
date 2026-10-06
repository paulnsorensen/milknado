"""Atomic execution-group provider identity update."""

from __future__ import annotations

import sqlite3


def bind_execution_group_provider(
    conn: sqlite3.Connection, group_id: str, provider_id: str
) -> None:
    if not provider_id:
        raise ValueError("provider identity must be nonempty")
    updated = conn.execute(
        "UPDATE execution_groups SET provider_session_id = ? WHERE id = ? "
        + "AND (provider_session_id IS NULL OR provider_session_id = ?)",
        (provider_id, group_id, provider_id),
    )
    if updated.rowcount != 1:
        raise ValueError("execution group provider identity changed")
