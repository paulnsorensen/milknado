"""Curd C (#297): graph self-heal — reconnect, quarantine, automigrate, close audit."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from milknado.domains.graph import MikadoGraph, _persistence

_GRAPH_LOG = "milknado.domains.graph.graph"


def test_closed_connection_reopens_on_next_access(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    """The #297 repro: graph.close() then get_execution_overview() must heal."""
    graph = MikadoGraph(tmp_path / "g.db")
    graph.add_node("alpha")
    graph.close()
    assert graph._close_stack is not None, "close() must record the calling stack"

    with caplog.at_level(logging.WARNING, logger=_GRAPH_LOG):
        _root, descriptions, _available = graph.get_execution_overview([1])
    assert descriptions == {1: "alpha"}

    warnings = [r for r in caplog.records if "reopened unexpectedly" in r.message]
    assert warnings, "unexpected reopen must log a loud warning"
    text = warnings[0].message
    assert "close stack:" in text and "heal stack:" in text

    # Reopened connection re-applied the pragmas.
    assert graph._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert graph._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert graph._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    graph.close()


def test_connection_killed_under_live_graph_heals(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    """The 03:47 lifetime-bug shape: the conn dies without close() being called."""
    graph = MikadoGraph(tmp_path / "g.db")
    graph.add_node("beta")
    graph._raw_conn.close()  # simulates the bug: closed behind the graph's back

    with caplog.at_level(logging.WARNING, logger=_GRAPH_LOG):
        nodes = graph.get_all_nodes()
    assert [n.description for n in nodes] == ["beta"]
    warnings = [r for r in caplog.records if "connection died without close()" in r.message]
    assert warnings
    assert "(no close() recorded)" in warnings[0].message
    graph.close()


def test_inflight_statement_at_close_time_fails_loud(tmp_path: Path) -> None:
    """No transparent retry: the statement on the dead conn raises its original
    error, and stays dead even after the graph heals itself for the NEXT op."""
    graph = MikadoGraph(tmp_path / "g.db")
    raw = graph._raw_conn
    graph.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        raw.execute("SELECT 1")
    graph.get_all_nodes()  # heals the graph's next access
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        raw.execute("SELECT 1")
    graph.close()


def test_corrupt_db_quarantined_and_recreated(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    db = tmp_path / "g.db"
    db.write_bytes(b"garbage-not-sqlite" * 64)
    wal = Path(f"{db}-wal")
    shm = Path(f"{db}-shm")
    wal.write_bytes(b"wal-garbage")
    shm.write_bytes(b"shm-garbage")

    with caplog.at_level(logging.WARNING, logger=_GRAPH_LOG):
        graph = MikadoGraph(db)

    moved = sorted(tmp_path.glob("*.corrupt-*"))
    assert len(moved) == 3, f"db + -wal + -shm must all quarantine, got {moved}"
    # The quarantine must happen BEFORE the corrupt handle closes — close()
    # on the last connection makes sqlite delete the -wal/-shm sidecars, so a
    # close-first implementation would move only 1 file, not 3. (Sidecar
    # contents are not asserted: sqlite rewrites -shm on open.)
    moved_by_suffix = {p.name.split(".corrupt-")[0].removeprefix("g.db"): p for p in moved}
    assert moved_by_suffix[""].read_bytes() == b"garbage-not-sqlite" * 64
    assert any("quarantined" in r.message for r in caplog.records)

    graph.add_node("fresh start")
    assert [n.description for n in graph.get_all_nodes()] == ["fresh start"]
    graph.close()


def test_fresh_db_migrates_to_current_version(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    version = graph._conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == _persistence.SCHEMA_VERSION
    tables = {r[0] for r in graph._conn.execute("SELECT name FROM sqlite_master").fetchall()}
    assert "node_reviews" in tables
    graph.close()


def _make_v1_db(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    _persistence.create_tables(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    conn.close()


def test_stale_db_automigrates_on_open(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    _make_v1_db(db)
    graph = MikadoGraph(db)
    version = graph._conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == _persistence.SCHEMA_VERSION
    node = graph.add_node("migrated")
    graph.insert_node_review(node.id, 1, "reject", "findings", "2026-07-23T00:00:00Z")
    rows = graph.node_reviews_for_node(node.id)
    assert [r["verdict"] for r in rows] == ["reject"]
    graph.close()


def test_migration_failure_rolls_back_and_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "g.db"
    _make_v1_db(db)
    monkeypatch.setattr(
        _persistence,
        "MIGRATIONS",
        [
            (2, "CREATE TABLE migration_probe (id INTEGER)"),
            (3, "SELEC bogus FROM nowhere"),
        ],
    )
    with pytest.raises(sqlite3.Error):
        MikadoGraph(db)
    conn = sqlite3.connect(str(db))
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    probe = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'migration_probe'"
    ).fetchone()
    conn.close()
    assert version == 0, "a failed batch must not stamp user_version"
    assert probe is None, "the earlier statement in the failed batch must roll back"


def test_close_is_idempotent(tmp_path: Path) -> None:
    """A second close() is a no-op: no crash, no heal, no clobbered close stack."""
    graph = MikadoGraph(tmp_path / "g.db")
    graph.add_node("twice closed")
    graph.close()
    recorded_stack = graph._close_stack
    assert recorded_stack is not None, "the close that closes must record its stack"
    graph.close()
    assert graph._closed is True
    assert graph._raw_conn is None
    assert graph._close_stack is recorded_stack, (
        "a redundant close must not clobber the original close stack — "
        "the heal warning names that culprit site"
    )
    # Next access still heals normally after the double close.
    assert [n.description for n in graph.get_all_nodes()] == ["twice closed"]
    graph.close()


def test_corrupt_db_without_sidecars_quarantines_db_only(tmp_path: Path) -> None:
    """No -wal/-shm on disk → only the db file is moved aside; fresh db still works."""
    db = tmp_path / "g.db"
    db.write_bytes(b"garbage" * 32)

    graph = MikadoGraph(db)

    moved = list(tmp_path.glob("*.corrupt-*"))
    assert len(moved) == 1
    assert moved[0].name.startswith("g.db.corrupt-")
    assert moved[0].read_bytes() == b"garbage" * 32
    graph.add_node("recovered")
    assert [n.description for n in graph.get_all_nodes()] == ["recovered"]
    graph.close()


def test_reopen_of_current_db_is_noop_migration(tmp_path: Path) -> None:
    """Reopening an already-migrated db must not re-apply or fail: migrate() is
    a no-op at SCHEMA_VERSION, and existing rows survive the reopen."""
    db = tmp_path / "g.db"
    graph = MikadoGraph(db)
    node = graph.add_node("persistent")
    graph.insert_node_review(node.id, 1, "reject", "f", "2026-07-23T00:00:00Z")
    graph.close()

    reopened = MikadoGraph(db)
    assert reopened._conn.execute("PRAGMA user_version").fetchone()[0] == (
        _persistence.SCHEMA_VERSION
    )
    rows = reopened.node_reviews_for_node(node.id)
    assert [r["verdict"] for r in rows] == ["reject"]
    reopened.close()
