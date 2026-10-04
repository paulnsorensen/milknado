"""SQLite connection setup and corrupt-database quarantine for the graph."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import milknado.domains.graph._persistence as _persistence

_logger = logging.getLogger(__name__)


def quick_check(conn: sqlite3.Connection) -> str:
    row = cast(object, conn.execute("PRAGMA quick_check").fetchone())
    values = cast(tuple[object, ...], row)
    return cast(str, values[0])


def _quarantine(db_path: Path) -> list[Path]:
    # Microsecond resolution prevents two quarantines from overwriting evidence.
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")
    moved: list[Path] = []
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{db_path}{suffix}")
        if candidate.exists():
            target = Path(f"{candidate}.corrupt-{ts}")
            _ = candidate.rename(target)
            moved.append(target)
    return moved


def open_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        check = quick_check(conn)
    except sqlite3.DatabaseError:
        check = "error"
    if check != "ok":
        # Quarantine before close; closing the last handle can remove WAL sidecars.
        moved = _quarantine(db_path)
        # The corrupt file is quarantined. A close failure cannot stop recreation.
        with suppress(sqlite3.Error):
            conn.close()
        _logger.warning(
            "database failed PRAGMA quick_check (%s); quarantined %s to %s; recreating fresh",
            check,
            db_path,
            [str(p) for p in moved],
        )
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ = conn.execute("PRAGMA journal_mode=WAL")
    _ = conn.execute("PRAGMA foreign_keys=ON")
    # Wait for other graph writers across processes.
    _ = conn.execute("PRAGMA busy_timeout=5000")
    try:
        _persistence.create_tables(conn)
        _persistence.migrate(conn)
    except Exception:
        # Keep the schema error if cleanup also fails.
        with suppress(sqlite3.Error):
            conn.close()
        raise
    return conn
