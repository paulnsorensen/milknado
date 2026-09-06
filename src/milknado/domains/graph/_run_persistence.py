"""Run, message, and review persistence for the graph domain."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TypedDict, cast

from milknado.domains.common import RunResult
from milknado.domains.graph._sqlite_rows import as_tuple as _as_tuple
from milknado.domains.graph._sqlite_rows import fetchall, fetchone

_logger = logging.getLogger(__name__)


class _RunRow(TypedDict):
    run_id: str
    node_id: int
    status: str
    pid: int | None
    log_path: str
    started_at: str
    ended_at: str | None
    timed_out: int
    exit_code: int | None
    error: str | None
    timeout_seconds: int | None
    detail: str | None
    rebased: int | None


class RunRecord(TypedDict):
    run_id: str
    node_id: int
    status: str
    pid: int | None
    log_path: str
    started_at: str
    ended_at: str | None
    timed_out: bool
    exit_code: int | None
    error: str | None
    timeout_seconds: int | None
    detail: str | None
    rebased: bool | None


class NodeReviewRecord(TypedDict):
    node_id: int
    round: int
    verdict: str
    findings: str
    created_at: str


_MAX_RETAINED_RUN_MESSAGES = 1000
_MAX_RUN_MESSAGE_BYTES = 64 * 1024
_RUN_MESSAGE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
_RUN_STATUS_RUNNING = "running"  # sole owner: runs.status compares case-sensitively


def run_row_to_dict(row: sqlite3.Row) -> RunRecord:
    """Serialize a runs row to the sidecar-shaped dict callers consumed before.

    `timed_out` / `rebased` are stored as INTEGER and re-hydrated to bool/None so
    a poll payload reads the same as the old JSON sidecar (which carried bools).
    """
    raw = cast(_RunRow, cast(object, row))
    rebased = raw["rebased"]
    return {
        "run_id": raw["run_id"],
        "node_id": raw["node_id"],
        "status": raw["status"],
        "pid": raw["pid"],
        "log_path": raw["log_path"],
        "started_at": raw["started_at"],
        "ended_at": raw["ended_at"],
        "timed_out": bool(raw["timed_out"]),
        "exit_code": raw["exit_code"],
        "error": raw["error"],
        "timeout_seconds": raw["timeout_seconds"],
        "detail": raw["detail"],
        "rebased": bool(rebased) if rebased is not None else None,
    }


def start_run(
    conn: sqlite3.Connection,
    run_id: str,
    node_id: int,
    log_path: str,
    started_at: str,
    timeout_seconds: int | None,
    pid: int | None = None,
) -> None:
    """INSERT a status='running' run row (replaces the initial sidecar write)."""
    _ = conn.execute(
        "INSERT INTO runs "
        + "(run_id, node_id, status, pid, log_path, started_at, timeout_seconds, timed_out) "
        + "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
        (run_id, node_id, _RUN_STATUS_RUNNING, pid, log_path, started_at, timeout_seconds),
    )
    conn.commit()


def finish_run(conn: sqlite3.Connection, run_id: str, result: RunResult) -> bool:
    """Write one terminal result; return whether this fenced write won."""
    cur = conn.execute(
        "UPDATE runs SET status = ?, exit_code = ?, timed_out = ?, ended_at = ?, "
        + "error = ?, detail = ?, rebased = ? WHERE run_id = ? AND status = ?",
        (
            result.status,
            result.exit_code,
            1 if result.timed_out else 0,
            result.ended_at,
            result.error,
            result.detail,
            None if result.rebased is None else (1 if result.rebased else 0),
            run_id,
            _RUN_STATUS_RUNNING,
        ),
    )
    conn.commit()
    won = cur.rowcount == 1
    if not won:
        _logger.warning(
            "finish_run dropped late terminal write for run %s (status=%s)",
            run_id,
            result.status,
        )
    return won


def set_run_pid(conn: sqlite3.Connection, run_id: str, pid: int) -> None:
    """Gated on status='running' so a runner that already wrote its terminal state
    is never clobbered back toward running (mirrors the old read-then-write
    guard the sidecar path used).
    """
    _ = conn.execute(
        "UPDATE runs SET pid = ? WHERE run_id = ? AND status = ?",
        (pid, run_id, _RUN_STATUS_RUNNING),
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> RunRecord | None:
    row = fetchone(conn, "SELECT * FROM runs WHERE run_id = ?", (run_id,))
    return run_row_to_dict(row) if row else None


def runs_for_node(
    conn: sqlite3.Connection,
    node_id: int,
    *,
    terminal_only: bool = False,
    run_id: str | None = None,
) -> list[RunRecord]:
    """Return this node's runs as dicts (replaces find_terminal_runs_for_node).

    Callers still apply fence-before-latest: pass run_id to filter to the fence
    BEFORE calling latest_terminal_run, so a stale run with a later ended_at
    cannot mask the owner.
    """
    sql = "SELECT * FROM runs WHERE node_id = ?"
    params: list[object] = [node_id]
    if terminal_only:
        sql += " AND status IN ('done', 'failed')"
    if run_id is not None:
        sql += " AND run_id = ?"
        params.append(run_id)
    rows = fetchall(conn, sql, params)
    return [run_row_to_dict(row) for row in rows]


def recent_runs(conn: sqlite3.Connection, limit: int) -> list[RunRecord]:
    rows = fetchall(conn, "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (max(limit, 0),))
    return [run_row_to_dict(row) for row in rows]


def _prune_run_messages(conn: sqlite3.Connection, created_at: str) -> None:
    cutoff = datetime.fromisoformat(created_at).timestamp() - _RUN_MESSAGE_MAX_AGE_SECONDS
    cutoff_at = datetime.fromtimestamp(cutoff, UTC).isoformat()
    terminal = "SELECT run_id FROM runs WHERE status != ?"
    _ = conn.execute(
        f"DELETE FROM run_messages WHERE run_id IN ({terminal}) AND created_at < ?",
        (_RUN_STATUS_RUNNING, cutoff_at),
    )
    _ = conn.execute(
        "DELETE FROM run_messages WHERE (run_id, seq) IN ("
        + f"SELECT run_id, seq FROM run_messages WHERE run_id IN ({terminal}) "
        + "ORDER BY created_at DESC, seq DESC LIMIT -1 OFFSET ?)",
        (_RUN_STATUS_RUNNING, _MAX_RETAINED_RUN_MESSAGES),
    )


def deposit_run_message(
    conn: sqlite3.Connection, run_id: str, role: str, body: str, created_at: str
) -> int:
    """Append a bounded run message and prune terminal history."""
    body = body.encode()[:_MAX_RUN_MESSAGE_BYTES].decode(errors="ignore")
    _prune_run_messages(conn, created_at)
    row = fetchone(
        conn,
        "INSERT INTO run_messages (run_id, seq, role, body, created_at) "
        + "SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ? FROM run_messages "
        + "WHERE run_id = ? RETURNING seq",
        (run_id, role, body, created_at, run_id),
    )
    if row is None:
        raise RuntimeError("deposit_run_message INSERT returned no sequence")
    conn.commit()
    return cast(int, _as_tuple(row)[0])


def deposit_review_verdict(
    conn: sqlite3.Connection, run_id: str, verdict: str, findings: str, created_at: str
) -> int:
    """Persist a verdict and atomically mark it terminal only for an active review run."""
    with conn:
        row = fetchone(
            conn,
            "INSERT INTO run_messages (run_id, seq, role, body, created_at) "
            + "SELECT ?, COALESCE(MAX(seq), 0) + 1, 'review', ?, ? "
            + "FROM run_messages WHERE run_id = ? RETURNING seq",
            (run_id, f"{verdict}\n{findings}", created_at, run_id),
        )
        if row is None:
            raise RuntimeError("deposit_review_verdict INSERT returned no sequence")
        terminal = fetchone(
            conn,
            "SELECT 1 FROM runs JOIN nodes ON nodes.id = runs.node_id "
            + "WHERE runs.run_id = ? AND runs.status = ? AND nodes.flavor = 'review'",
            (run_id, _RUN_STATUS_RUNNING),
        )
        if terminal is not None:
            _ = conn.execute(
                "INSERT INTO run_messages (run_id, seq, role, body, created_at) "
                + "SELECT ?, COALESCE(MAX(seq), 0) + 1, 'review_terminal', ?, ? "
                + "FROM run_messages WHERE run_id = ?",
                (run_id, verdict, created_at, run_id),
            )
    return cast(int, _as_tuple(row)[0])


def insert_node_review(
    conn: sqlite3.Connection,
    node_id: int,
    round_number: int | None,
    verdict: str,
    findings: str,
    created_at: str,
) -> int:
    with conn:
        if round_number is None:
            sql = (
                "INSERT INTO node_reviews (node_id, round, verdict, findings, created_at) "
                "SELECT ?, COALESCE(MAX(round), 0) + 1, ?, ?, ? "
                "FROM node_reviews WHERE node_id = ? RETURNING round"
            )
            row = fetchone(conn, sql, (node_id, verdict, findings, created_at, node_id))
            if row is None:
                raise RuntimeError("node review insert returned no row")
            return cast(int, _as_tuple(row)[0])
        _ = conn.execute(
            "INSERT INTO node_reviews (node_id, round, verdict, findings, created_at) "
            + "VALUES (?, ?, ?, ?, ?)",
            (node_id, round_number, verdict, findings, created_at),
        )
    return round_number


def node_reviews_for_node(conn: sqlite3.Connection, node_id: int) -> list[NodeReviewRecord]:
    rows = fetchall(
        conn,
        "SELECT node_id, round, verdict, findings, created_at "
        + "FROM node_reviews WHERE node_id = ? ORDER BY round",
        (node_id,),
    )
    return [
        {
            "node_id": cast(int, values[0]),
            "round": cast(int, values[1]),
            "verdict": cast(str, values[2]),
            "findings": cast(str, values[3]),
            "created_at": cast(str, values[4]),
        }
        for values in (_as_tuple(row) for row in rows)
    ]


def latest_run_message(conn: sqlite3.Connection, run_id: str, role: str) -> str | None:
    row = fetchone(
        conn,
        "SELECT body FROM run_messages WHERE run_id = ? AND role = ? "
        + "ORDER BY seq DESC LIMIT 1",
        (run_id, role),
    )
    return cast(str, _as_tuple(row)[0]) if row else None
