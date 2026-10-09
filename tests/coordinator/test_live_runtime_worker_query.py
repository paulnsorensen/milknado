"""Connection-scoped worker evidence protects coordinator turn fences."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from milknado.domains.coordinator.turn_fences import TurnFence, clear_verified_fence
from milknado.domains.graph import MikadoGraph, live_runtime_workers


def _insert_worker(
    conn: sqlite3.Connection, invocation_id: str, turn_id: str, ended_at: str | None = None
) -> None:
    _ = conn.execute(
        "INSERT INTO run_workers "
        + "(invocation_id, runtime_run_id, supervisor_pid, supervisor_start_token, "
        + "pid, pgid, start_token, started_at, ended_at) "
        + "VALUES (?, ?, 999999, 1.0, 2345, 2345, 1.0, 'now', ?)",
        (invocation_id, turn_id, ended_at),
    )


def test_live_runtime_workers_sees_uncommitted_worker_on_own_connection(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with sqlite3.connect(graph.db_path) as conn:
        conn.row_factory = sqlite3.Row
        _ = conn.execute("BEGIN IMMEDIATE")
        _insert_worker(conn, "live", "turn")
        _insert_worker(conn, "other", "unrelated")
        _insert_worker(conn, "ended", "turn", "later")
        assert [worker.invocation_id for worker in live_runtime_workers(conn, "turn")] == ["live"]
    graph.close()


def test_clear_verified_fence_keeps_live_worker_and_ignores_other_rows(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    fence = TurnFence("turn", "coordinator", "scope", 999999, 1.0)
    with sqlite3.connect(graph.db_path) as conn:
        conn.row_factory = sqlite3.Row
        _ = conn.execute(
            "INSERT INTO coordinator_turn_launches "
            + "(command_id, coordinator_id, scope_kind, scope_id, state, "
            + "supervisor_pid, supervisor_start_token) "
            + "VALUES ('turn', 'session', 'coordinator', 'scope', 'submitted', 999999, 1.0)"
        )
        _insert_worker(conn, "live", "turn")
        conn.commit()
        assert not clear_verified_fence(conn, fence)
        assert (
            conn.execute(
                "SELECT state FROM coordinator_turn_launches WHERE command_id = 'turn'"
            ).fetchone()[0]
            == "submitted"
        )
        _ = conn.execute("UPDATE run_workers SET ended_at = 'later' WHERE invocation_id = 'live'")
        _insert_worker(conn, "other", "unrelated")
        conn.commit()
        assert clear_verified_fence(conn, fence)
        assert (
            conn.execute(
                "SELECT state FROM coordinator_turn_launches WHERE command_id = 'turn'"
            ).fetchone()[0]
            == "unknown"
        )
    graph.close()
