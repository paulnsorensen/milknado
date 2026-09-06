"""Curd C (#297): graph self-heal — reconnect, quarantine, automigrate, close audit."""

from __future__ import annotations

# These tests intentionally exercise protected graph recovery state.
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from typing_extensions import override

from milknado.domains.execution import get_execution_overview
from milknado.domains.graph import MikadoGraph, _persistence
from tests.graph_helpers import (
    graph_close_stack,
    graph_conn,
    graph_is_closed,
    graph_raw_conn,
)

_GRAPH_LOG = "milknado.domains.graph.graph"


def test_closed_connection_reopens_on_next_access(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The #297 repro: graph.close() then execution reads its snapshot must heal."""
    graph = MikadoGraph(tmp_path / "g.db")
    _ = graph.add_node("alpha")
    graph.close()
    assert graph_close_stack(graph) is not None, "close() must record the calling stack"

    with caplog.at_level(logging.WARNING, logger=_GRAPH_LOG):
        _root, descriptions, _available = get_execution_overview(graph, [1])
    assert descriptions == {1: "alpha"}

    warnings = [r for r in caplog.records if "reopened unexpectedly" in r.message]
    assert warnings, "unexpected reopen must log a loud warning"
    text = warnings[0].message
    assert "close stack:" in text and "heal stack:" in text

    # Reopened connection re-applied the pragmas.
    assert graph_conn(graph).execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert graph_conn(graph).execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert graph_conn(graph).execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    graph.close()


def test_connection_killed_under_live_graph_heals(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The 03:47 lifetime-bug shape: the conn dies without close() being called."""
    graph = MikadoGraph(tmp_path / "g.db")
    _ = graph.add_node("beta")
    raw = graph_raw_conn(graph)
    assert raw is not None
    raw.close()  # simulates the bug: closed behind the graph's back

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
    raw = graph_raw_conn(graph)
    assert raw is not None
    graph.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        _ = raw.execute("SELECT 1")
    _ = graph.get_all_nodes()  # heals the graph's next access
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        _ = raw.execute("SELECT 1")
    graph.close()


def test_corrupt_db_quarantined_and_recreated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    db = tmp_path / "g.db"
    _ = db.write_bytes(b"garbage-not-sqlite" * 64)
    wal = Path(f"{db}-wal")
    shm = Path(f"{db}-shm")
    _ = wal.write_bytes(b"wal-garbage")
    _ = shm.write_bytes(b"shm-garbage")

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

    _ = graph.add_node("fresh start")
    assert [n.description for n in graph.get_all_nodes()] == ["fresh start"]
    graph.close()


def test_fresh_db_migrates_to_current_version(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    version = cast(int, graph_conn(graph).execute("PRAGMA user_version").fetchone()[0])
    assert version == _persistence.SCHEMA_VERSION
    rows = cast(
        list[tuple[str]],
        graph_conn(graph).execute("SELECT name FROM sqlite_master").fetchall(),
    )
    tables = {r[0] for r in rows}
    assert "node_reviews" in tables
    graph.close()


def _make_v1_db(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    _persistence.create_tables(conn)
    assert cast(int, conn.execute("PRAGMA user_version").fetchone()[0]) == 0
    conn.close()


def test_stale_db_automigrates_on_open(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    _make_v1_db(db)
    graph = MikadoGraph(db)
    version = cast(int, graph_conn(graph).execute("PRAGMA user_version").fetchone()[0])
    assert version == _persistence.SCHEMA_VERSION
    node = graph.add_node("migrated")
    _ = graph.insert_node_review(node.id, "reject", "findings", "2026-07-23T00:00:00Z")
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
        _ = MikadoGraph(db)
    conn = sqlite3.connect(str(db))
    version = cast(int, conn.execute("PRAGMA user_version").fetchone()[0])
    probe = cast(
        tuple[object] | None,
        conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'migration_probe'").fetchone(),
    )
    conn.close()
    assert version == 0, "a failed batch must not stamp user_version"
    assert probe is None, "the earlier statement in the failed batch must roll back"


def test_close_is_idempotent(tmp_path: Path) -> None:
    """A second close() is a no-op: no crash, no heal, no clobbered close stack."""
    graph = MikadoGraph(tmp_path / "g.db")
    _ = graph.add_node("twice closed")
    graph.close()
    recorded_stack = graph_close_stack(graph)
    assert recorded_stack is not None, "the close that closes must record its stack"
    graph.close()
    assert graph_is_closed(graph) is True
    assert graph_raw_conn(graph) is None
    assert graph_close_stack(graph) is recorded_stack, (
        "a redundant close must not clobber the original close stack — "
        "the heal warning names that culprit site"
    )
    # Next access still heals normally after the double close.
    assert [n.description for n in graph.get_all_nodes()] == ["twice closed"]
    graph.close()


def test_corrupt_db_without_sidecars_quarantines_db_only(tmp_path: Path) -> None:
    """No -wal/-shm on disk → only the db file is moved aside; fresh db still works."""
    db = tmp_path / "g.db"
    _ = db.write_bytes(b"garbage" * 32)

    graph = MikadoGraph(db)

    moved = list(tmp_path.glob("*.corrupt-*"))
    assert len(moved) == 1
    assert moved[0].name.startswith("g.db.corrupt-")
    assert moved[0].read_bytes() == b"garbage" * 32
    _ = graph.add_node("recovered")
    assert [n.description for n in graph.get_all_nodes()] == ["recovered"]
    graph.close()


def test_reopen_of_current_db_is_noop_migration(tmp_path: Path) -> None:
    """Reopening an already-migrated db must not re-apply or fail: migrate() is
    a no-op at SCHEMA_VERSION, and existing rows survive the reopen."""
    db = tmp_path / "g.db"
    graph = MikadoGraph(db)
    node = graph.add_node("persistent")
    _ = graph.insert_node_review(node.id, "reject", "f", "2026-07-23T00:00:00Z")
    graph.close()

    reopened = MikadoGraph(db)
    assert graph_conn(reopened).execute("PRAGMA user_version").fetchone()[0] == (
        _persistence.SCHEMA_VERSION
    )
    rows = reopened.node_reviews_for_node(node.id)
    assert [r["verdict"] for r in rows] == ["reject"]
    reopened.close()


def _review_count(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    count = cast(int, conn.execute("SELECT COUNT(*) FROM node_reviews").fetchone()[0])
    conn.close()
    return count


def test_delete_node_cleans_node_reviews(tmp_path: Path) -> None:
    """node_reviews.node_id is a bare REFERENCES FK (no ON DELETE action) with
    PRAGMA foreign_keys=ON: delete_node must remove the dependent review rows
    itself or the delete raises IntegrityError."""
    db = tmp_path / "g.db"
    graph = MikadoGraph(db)
    node = graph.add_node("reviewed")
    _ = graph.insert_node_review(node.id, "reject", "findings", "2026-07-23T00:00:00Z")

    deleted = graph.delete_node(node.id)

    assert deleted == 1
    assert _review_count(db) == 0, "delete_node must clear the node's review rows"
    graph.close()


def test_drop_all_cleans_node_reviews(tmp_path: Path) -> None:
    """drop_all wipes every table; node_reviews must be cleared before nodes or
    the FK blocks the bulk delete."""
    db = tmp_path / "g.db"
    graph = MikadoGraph(db)
    node = graph.add_node("reviewed")
    _ = graph.insert_node_review(node.id, "approve", "", "2026-07-23T00:00:00Z")

    count = graph.drop_all()

    assert count == 1
    assert _review_count(db) == 0, "drop_all must clear node_reviews"
    graph.close()


def test_open_failure_closes_connection_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If create_tables/migrate raises during _open, the fresh handle must be
    closed (best-effort) and the ORIGINAL exception propagate unchanged."""
    original = RuntimeError("schema boom")

    def _boom(_conn: sqlite3.Connection) -> None:
        raise original

    monkeypatch.setattr(_persistence, "create_tables", _boom)
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def _tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connector = cast(Callable[..., sqlite3.Connection], real_connect)
        conn = connector(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _tracking_connect)
    with pytest.raises(RuntimeError) as exc_info:
        _ = MikadoGraph(tmp_path / "g.db")
    assert exc_info.value is original
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        _ = opened[0].execute("SELECT 1")


class _FlakyCloseConn(sqlite3.Connection):
    """The first close() across all instances raises sqlite3.Error; the rest succeed."""

    _armed: bool = True

    @override
    def close(self) -> None:
        cls = type(self)
        if cls._armed:
            cls._armed = False
            raise sqlite3.Error("close boom")
        super().close()


def _use_flaky_close_conns(monkeypatch: pytest.MonkeyPatch) -> list[_FlakyCloseConn]:
    _FlakyCloseConn._armed = True  # pyright: ignore[reportPrivateUsage]
    opened: list[_FlakyCloseConn] = []
    real_connect = sqlite3.connect

    def _connect_with_factory(*args: object, **kwargs: object) -> _FlakyCloseConn:
        connector = cast(Callable[..., _FlakyCloseConn], real_connect)
        conn = connector(*args, factory=_FlakyCloseConn, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _connect_with_factory)
    return opened


def test_quarantine_swallows_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quarantine path's close is best-effort: a failing close() must not
    stop the quarantine + fresh-recreate heal."""
    db = tmp_path / "g.db"
    _ = db.write_bytes(b"garbage-not-sqlite" * 64)
    opened = _use_flaky_close_conns(monkeypatch)

    graph = MikadoGraph(db)

    moved = list(tmp_path.glob("*.corrupt-*"))
    assert len(moved) == 1, "quarantine must still move the corrupt db aside"
    _ = graph.add_node("healed")
    assert [n.description for n in graph.get_all_nodes()] == ["healed"]
    graph.close()
    for conn in opened:
        conn.close()  # second close succeeds; no leaked handles


def test_open_failure_swallows_close_failure_and_reraises_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When create_tables raises AND the cleanup close() also fails, the
    ORIGINAL exception must still propagate unchanged."""
    original = RuntimeError("schema boom")

    def _boom(_conn: sqlite3.Connection) -> None:
        raise original

    monkeypatch.setattr(_persistence, "create_tables", _boom)
    opened = _use_flaky_close_conns(monkeypatch)

    with pytest.raises(RuntimeError) as exc_info:
        _ = MikadoGraph(tmp_path / "g.db")
    assert exc_info.value is original
    for conn in opened:
        conn.close()  # second close succeeds; no leaked handles


def test_close_before_first_use(tmp_path: Path) -> None:
    """close() with no prior _conn access is a clean no-surprise shutdown."""
    graph = MikadoGraph(tmp_path / "g.db")
    graph.close()
    assert graph_is_closed(graph) is True
