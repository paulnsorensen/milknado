"""Shared sqlite3.Row helpers for the graph domain's persistence modules."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import cast


def fetchone(
    conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()
) -> sqlite3.Row | None:
    return cast(sqlite3.Row | None, conn.execute(sql, params).fetchone())


def fetchall(
    conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()
) -> list[sqlite3.Row]:
    return cast(list[sqlite3.Row], conn.execute(sql, params).fetchall())


def as_tuple(row: sqlite3.Row) -> tuple[object, ...]:
    return cast(tuple[object, ...], cast(object, row))
