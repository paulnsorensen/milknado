"""Tests for graph/_persistence.py — schema, row serialization, plan_state, drop_all."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from milknado.domains.graph import MikadoGraph
from milknado.domains.graph._persistence import (
    create_tables,
    ensure_schema,
    get_spec_hash,
    row_to_node,
    set_dispatched_at,
    set_spec_hash,
)


@pytest.fixture()
def conn(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    c = sqlite3.connect(str(tmp_path / "p.db"))
    c.row_factory = sqlite3.Row
    create_tables(c)
    ensure_schema(c)
    yield c
    c.close()


@pytest.fixture()
def graph(tmp_path: Path) -> Generator[MikadoGraph, None, None]:
    g = MikadoGraph(tmp_path / "g.db")
    yield g
    g.close()


class TestCreateTables:
    def test_tables_created(self, conn: sqlite3.Connection) -> None:
        names = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {
            "nodes",
            "edges",
            "file_ownership",
            "plan_state",
            "batch_plans",
            "runs",
            "run_messages",
        } <= names

    def test_runs_node_status_index_created(self, conn: sqlite3.Connection) -> None:
        """idx_runs_node_status backs runs_for_node's WHERE node_id (+status) — it is
        part of the schema contract, so a dropped CREATE INDEX must fail a test."""
        indexes = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "idx_runs_node_status" in indexes


class TestPlanState:
    def test_set_and_get_spec_hash(self, conn: sqlite3.Connection) -> None:
        set_spec_hash(conn, "abc123")
        assert get_spec_hash(conn) == "abc123"

    def test_get_spec_hash_empty_returns_none(self, conn: sqlite3.Connection) -> None:
        assert get_spec_hash(conn) is None

    def test_set_spec_hash_upserts(self, conn: sqlite3.Connection) -> None:
        set_spec_hash(conn, "first")
        set_spec_hash(conn, "second")
        assert get_spec_hash(conn) == "second"

    def test_graph_set_get_spec_hash(self, graph: MikadoGraph) -> None:
        graph.set_spec_hash("deadbeef")
        assert graph.get_spec_hash() == "deadbeef"

    def test_graph_get_spec_hash_none_before_set(self, graph: MikadoGraph) -> None:
        assert graph.get_spec_hash() is None


class TestDropAll:
    def test_drop_all_returns_node_count(self, graph: MikadoGraph) -> None:
        graph.add_node("one")
        graph.add_node("two")
        count = graph.drop_all()
        assert count == 2

    def test_drop_all_clears_nodes(self, graph: MikadoGraph) -> None:
        graph.add_node("node")
        graph.drop_all()
        assert graph.get_all_nodes() == []

    def test_drop_all_clears_plan_state(self, graph: MikadoGraph) -> None:
        graph.set_spec_hash("abc")
        graph.drop_all()
        assert graph.get_spec_hash() is None

    def test_drop_all_on_empty_returns_zero(self, graph: MikadoGraph) -> None:
        assert graph.drop_all() == 0


class TestRunsRepo:
    """Direct boundary + status-transition coverage for the runs-table repo
    methods (start_run / finish_run / get_run / runs_for_node / recent_runs /
    set_run_pid). These back the dispatch reconcilers, so a regression in a
    re-hydration or a fence here surfaces as a misreported run state."""

    @staticmethod
    def _node(graph: MikadoGraph, desc: str = "task") -> int:
        return graph.add_node(desc).id

    def test_start_then_get_round_trips_running_row(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("r1", nid, "/logs/r1.log", "2026-01-01T00:00:00+00:00", 600, None)
        row = graph.get_run("r1")
        assert row is not None
        assert row["status"] == "running"
        assert row["node_id"] == nid
        assert row["log_path"] == "/logs/r1.log"
        assert row["timeout_seconds"] == 600
        assert row["pid"] is None
        assert row["ended_at"] is None
        # INTEGER default 0 must re-hydrate to a bool, not a raw 0.
        assert row["timed_out"] is False
        assert row["rebased"] is None

    def test_get_run_unknown_returns_none(self, graph: MikadoGraph) -> None:
        assert graph.get_run("nope") is None

    def test_finish_run_transitions_running_to_done(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("r1", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.finish_run(
            "r1",
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at="2026-01-01T00:01:00+00:00",
            detail="all-green",
        )
        row = graph.get_run("r1")
        assert row is not None
        assert row["status"] == "done"
        assert row["exit_code"] == 0
        assert row["ended_at"] == "2026-01-01T00:01:00+00:00"
        assert row["detail"] == "all-green"

    def test_finish_run_rehydrates_timed_out_and_rebased_bools(self, graph: MikadoGraph) -> None:
        """timed_out/rebased are stored as INTEGER; _run_row_to_dict must return
        real bools (and keep rebased=None distinct from rebased=False) so a poll
        payload matches the old JSON sidecar's bool shape."""
        nid = self._node(graph)
        graph.start_run("rt", nid, "/l", "2026-01-01T00:00:00+00:00", 5)
        graph.finish_run(
            "rt",
            status="failed",
            exit_code=-1,
            timed_out=True,
            ended_at="2026-01-01T00:00:10+00:00",
            rebased=False,
        )
        row = graph.get_run("rt")
        assert row is not None
        assert row["timed_out"] is True
        assert row["rebased"] is False, "rebased=False must stay False, not collapse to None"

    def test_finish_run_rebased_true_round_trips(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("rb", nid, "/l", "2026-01-01T00:00:00+00:00", 5)
        graph.finish_run(
            "rb",
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at="2026-01-01T00:00:10+00:00",
            rebased=True,
        )
        row = graph.get_run("rb")
        assert row is not None
        assert row["rebased"] is True

    def test_finish_run_does_not_clobber_terminal_status(self, graph: MikadoGraph) -> None:
        """First terminal write wins: a late worker finish landing after the run
        was already finalized (e.g. cancel's takeover write) must not flip the
        recorded outcome — an unconditional UPDATE would reintroduce the
        terminal-state clobber race the old sidecar flow guarded against."""
        nid = self._node(graph)
        graph.start_run("rc", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.finish_run(
            "rc",
            status="failed",
            exit_code=-1,
            timed_out=False,
            ended_at="2026-01-01T00:01:00+00:00",
            error="cancelled",
        )
        # The wedged worker finally finishes and lands its own terminal write.
        graph.finish_run(
            "rc",
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at="2026-01-01T00:02:00+00:00",
        )
        row = graph.get_run("rc")
        assert row is not None
        assert row["status"] == "failed", "late terminal write must not clobber the first"
        assert row["error"] == "cancelled"
        assert row["ended_at"] == "2026-01-01T00:01:00+00:00"

    def test_set_run_pid_writes_on_running(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("rp", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.set_run_pid("rp", 4321)
        row = graph.get_run("rp")
        assert row is not None
        assert row["pid"] == 4321

    def test_set_run_pid_no_op_on_terminal_run(self, graph: MikadoGraph) -> None:
        """The status='running' gate: a runner that already wrote its terminal
        state must never be clobbered back toward running by a late pid write.
        Drop the gate and this pid lands on a 'done' row — exactly the regression
        the sidecar read-then-write guard prevented."""
        nid = self._node(graph)
        graph.start_run("rt", nid, "/l", "2026-01-01T00:00:00+00:00", 600, pid=11)
        graph.finish_run(
            "rt",
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at="2026-01-01T00:00:10+00:00",
        )
        graph.set_run_pid("rt", 9999)
        row = graph.get_run("rt")
        assert row is not None
        assert row["pid"] == 11, "set_run_pid must not write a pid onto a terminal run"

    def test_runs_for_node_terminal_only_excludes_running(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("live", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.start_run("done", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.finish_run(
            "done",
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at="2026-01-01T00:00:10+00:00",
        )
        all_runs = {r["run_id"] for r in graph.runs_for_node(nid)}
        terminal = {r["run_id"] for r in graph.runs_for_node(nid, terminal_only=True)}
        assert all_runs == {"live", "done"}
        assert terminal == {"done"}, "terminal_only must drop the still-running run"

    def test_runs_for_node_isolates_node_id(self, graph: MikadoGraph) -> None:
        a = self._node(graph, "a")
        b = self._node(graph, "b")
        graph.start_run("ra", a, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.start_run("rb", b, "/l", "2026-01-01T00:00:00+00:00", 600)
        assert [r["run_id"] for r in graph.runs_for_node(a)] == ["ra"]

    def test_recent_runs_orders_by_started_at_desc(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("old", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.start_run("new", nid, "/l", "2026-01-02T00:00:00+00:00", 600)
        graph.start_run("mid", nid, "/l", "2026-01-01T12:00:00+00:00", 600)
        assert [r["run_id"] for r in graph.recent_runs(10)] == ["new", "mid", "old"]

    def test_recent_runs_honors_limit(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("old", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.start_run("new", nid, "/l", "2026-01-02T00:00:00+00:00", 600)
        assert [r["run_id"] for r in graph.recent_runs(1)] == ["new"]

    def test_recent_runs_zero_limit_returns_empty(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("r", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        assert graph.recent_runs(0) == []

    def test_deposit_run_message_increments_seq(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("r", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        assert graph.deposit_run_message("r", "result", "a", "2026-01-01T00:00:01+00:00") == 1
        assert graph.deposit_run_message("r", "result", "b", "2026-01-01T00:00:02+00:00") == 2

    def test_latest_run_message_is_role_scoped(self, graph: MikadoGraph) -> None:
        """latest_run_message must return the latest body for the asked role only —
        a later 'progress' row must not mask the 'result' the poll surfaces."""
        nid = self._node(graph)
        graph.start_run("r", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.deposit_run_message("r", "result", "the-deliverable", "2026-01-01T00:00:01+00:00")
        graph.deposit_run_message("r", "progress", "step-2", "2026-01-01T00:00:02+00:00")
        assert graph.latest_run_message("r", "result") == "the-deliverable"
        assert graph.latest_run_message("r", "progress") == "step-2"

    def test_latest_run_message_unknown_returns_none(self, graph: MikadoGraph) -> None:
        nid = self._node(graph)
        graph.start_run("r", nid, "/l", "2026-01-01T00:00:00+00:00", 600)
        assert graph.latest_run_message("r", "result") is None


class TestSetDispatchedAt:
    def test_set_dispatched_at_raises_for_missing_node(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            set_dispatched_at(conn, 9999)

    def test_set_dispatched_at_persists(self, graph: MikadoGraph) -> None:
        graph.add_node("task")
        graph.set_dispatched_at(1)
        node = graph.get_node(1)
        assert node is not None
        assert node.dispatched_at is not None


class TestSchemaEvolution:
    def test_ensure_schema_adds_missing_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        c.executescript("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parent_id INTEGER,
                worktree_path TEXT,
                branch_name TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE edges (
                parent_id INTEGER NOT NULL,
                child_id INTEGER NOT NULL,
                PRIMARY KEY (parent_id, child_id)
            );
            CREATE TABLE file_ownership (
                node_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                PRIMARY KEY (node_id, file_path)
            );
        """)
        c.execute(
            "INSERT INTO nodes (description, status, created_at) VALUES (?, ?, ?)",
            ("legacy", "pending", "2026-01-01T00:00:00+00:00"),
        )
        c.commit()
        ensure_schema(c)
        cols = {row[1] for row in c.execute("PRAGMA table_info(nodes)").fetchall()}
        assert "run_id" in cols
        assert "dispatched_at" in cols
        assert "completion_duration_seconds" in cols
        c.close()

    def test_ensure_schema_idempotent(self, conn: sqlite3.Connection) -> None:
        # Running ensure_schema twice should not raise
        ensure_schema(conn)
        ensure_schema(conn)


class TestRowToNode:
    def test_handles_missing_optional_columns(self, tmp_path: Path) -> None:
        """row_to_node gracefully handles rows without optional columns."""
        db_path = tmp_path / "sparse.db"
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        c.executescript("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parent_id INTEGER,
                worktree_path TEXT,
                branch_name TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
        """)
        c.execute(
            "INSERT INTO nodes (description, status, created_at) VALUES (?, ?, ?)",
            ("sparse node", "pending", "2026-01-01T00:00:00+00:00"),
        )
        c.commit()
        row = c.execute("SELECT * FROM nodes WHERE id = 1").fetchone()
        node = row_to_node(row)
        assert node.run_id is None
        assert node.dispatched_at is None
        assert node.completion_duration_seconds is None
        c.close()


class TestWalDurability:
    """Approach A (spec: run-state-storage): the WAL must fold into the main
    .db file so the documented `git add -f .milknado/milknado.db` persistence
    step cannot drop the uncheckpointed tail when a container is reclaimed."""

    def test_close_checkpoints_shared_wal_into_main_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "wal.db"
        # Long-lived writer, mirroring the detached ralph runner: commits a node
        # but stays open, so its frames sit in the shared -wal.
        runner = MikadoGraph(db_path)
        try:
            runner.add_node("written-by-runner")
            # Short-lived MCP-tool-call connection: open then close. Its close()
            # must checkpoint the shared WAL. A non-last-connection close does no
            # implicit checkpoint, so without the fix the frames stay in -wal.
            MikadoGraph(db_path).close()
            # Model `git add -f milknado.db` capturing ONLY the main db file.
            committed = tmp_path / "committed.db"
            shutil.copy(db_path, committed)
        finally:
            runner.close()
        fresh = sqlite3.connect(str(committed))
        try:
            count = fresh.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        finally:
            fresh.close()
        assert count == 1, "WAL tail lost: main db missing the runner's committed write"

    def test_busy_timeout_is_explicit(self, graph: MikadoGraph) -> None:
        """busy_timeout is pinned to 5000ms so concurrent writers (detached
        runner + server) wait rather than failing instantly — guards against a
        future connect() change silently dropping the wait window."""
        (timeout,) = graph._conn.execute("PRAGMA busy_timeout").fetchone()
        assert timeout == 5000

    def test_close_survives_checkpoint_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failed WAL checkpoint must never propagate out of close() — the
        connection still has to be released and the failure only logged, so a
        checkpoint error can't crash an MCP tool call mid-cleanup."""
        graph = MikadoGraph(tmp_path / "g.db")
        graph._conn.close()  # force the checkpoint PRAGMA to raise sqlite3.Error
        with caplog.at_level(logging.WARNING):
            graph.close()  # must not raise
        assert "WAL checkpoint on close failed" in caplog.text
