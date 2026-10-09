from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from milknado.app.watch import WatchSnapshotSource
from milknado.domains.coordinator import read_coordinator_status
from milknado.domains.graph import connect_readonly


def test_status_projection_keeps_missing_and_empty_tables_read_only(tmp_path: Path) -> None:
    db_path = tmp_path / "status.db"
    with sqlite3.connect(db_path) as writer:
        _ = writer.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, status TEXT NOT NULL)")
    source = WatchSnapshotSource(tmp_path, db_path)
    assert source.coordinator_status() == "No coordinator session is recorded."
    with sqlite3.connect(db_path) as writer:
        assert (
            writer.execute(
                "SELECT name FROM sqlite_master WHERE name = 'coordinator_sessions'"
            ).fetchone()
            is None
        )
        _ = writer.execute(
            "CREATE TABLE coordinator_sessions "
            + "(id TEXT PRIMARY KEY, goal_id INTEGER, provider TEXT, created_at TEXT)"
        )
        _ = writer.execute(
            "CREATE TABLE coordinator_events "
            + "(seq INTEGER PRIMARY KEY, session_id TEXT, kind TEXT, status TEXT)"
        )
    assert source.coordinator_status() == "No coordinator session is recorded."
    with connect_readonly(db_path) as reader:
        assert read_coordinator_status(reader) == ()
        with pytest.raises(sqlite3.OperationalError):
            _ = reader.execute("INSERT INTO nodes VALUES (1, 'todo')")


def test_status_projection_limits_order_and_uses_latest_recovery(tmp_path: Path) -> None:
    db_path = tmp_path / "status.db"
    with sqlite3.connect(db_path) as writer:
        _ = writer.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, status TEXT NOT NULL)")
        _ = writer.execute(
            "CREATE TABLE coordinator_sessions "
            + "(id TEXT PRIMARY KEY, goal_id INTEGER, provider TEXT, created_at TEXT)"
        )
        _ = writer.execute(
            "CREATE TABLE coordinator_events "
            + "(seq INTEGER PRIMARY KEY, session_id TEXT, kind TEXT, status TEXT)"
        )
        for index in range(12):
            _ = writer.execute("INSERT INTO nodes VALUES (?, ?)", (index, "ready"))
            _ = writer.execute(
                "INSERT INTO coordinator_sessions VALUES (?, ?, ?, ?)",
                (f"session-{index}", index, "codex", f"2026-10-08T00:00:{index:02d}Z"),
            )
        _ = writer.execute(
            "INSERT INTO coordinator_events (session_id, kind, status) VALUES (?, ?, ?)",
            ("session-11", "recovery", "old"),
        )
        _ = writer.execute(
            "INSERT INTO coordinator_events (session_id, kind, status) VALUES (?, ?, ?)",
            ("session-11", "recovery", "latest"),
        )
    with connect_readonly(db_path) as reader:
        rows = read_coordinator_status(reader)
    assert len(rows) == 10
    assert [row.goal_id for row in rows] == list(range(11, 1, -1))
    assert rows[0].recovery == "latest"
    assert rows[1].recovery is None
    assert WatchSnapshotSource(tmp_path, db_path).coordinator_status().splitlines()[0] == (
        "Goal 11 · codex · ready · recovery: latest"
    )
