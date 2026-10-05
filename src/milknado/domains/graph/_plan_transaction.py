from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import cast


@contextmanager
def plan_transaction(
    conn: sqlite3.Connection, lock: AbstractContextManager[object], expected_revision: int
) -> Generator[bool]:
    with lock, conn:
        _ = conn.execute("BEGIN IMMEDIATE")
        row = cast(
            tuple[int] | None,
            conn.execute("SELECT revision FROM graph_revision WHERE id = 1").fetchone(),
        )
        if row is None:
            raise RuntimeError("graph revision is unavailable")
        current = row[0] == expected_revision
        if current:
            _ = conn.execute("UPDATE graph_revision SET revision = revision + 1 WHERE id = 1")
        yield current
