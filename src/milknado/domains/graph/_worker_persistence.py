"""Durable loop-worker evidence and generation fences."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeAlias, cast

import msgspec

from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.graph._sqlite_rows import fetchall, fetchone

Descendant: TypeAlias = tuple[int, float, int]

CREATE_RUN_WORKERS = (
    "CREATE TABLE IF NOT EXISTS run_workers ("
    "invocation_id TEXT PRIMARY KEY, "
    "runtime_run_id TEXT NOT NULL, supervisor_pid INTEGER NOT NULL, "
    "supervisor_start_token REAL NOT NULL, "
    "graph_run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE, "
    "node_id INTEGER REFERENCES nodes(id) ON DELETE CASCADE, "
    "pid INTEGER NOT NULL, pgid INTEGER NOT NULL, start_token REAL NOT NULL, "
    "helper_pid INTEGER, helper_start_token REAL, "
    "helper_generation INTEGER NOT NULL DEFAULT -1, "
    "ready_generation INTEGER NOT NULL DEFAULT -1, "
    "snapshot_seq INTEGER NOT NULL DEFAULT 0, "
    "observation_owner TEXT, observation_seq INTEGER, "
    "observation_generation INTEGER, observation_pid INTEGER, observation_token REAL, "
    "descendants_json TEXT NOT NULL DEFAULT '[]', "
    "started_at TEXT NOT NULL, ended_at TEXT, "
    "CHECK ((graph_run_id IS NULL) = (node_id IS NULL)))"
)


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    runtime_run_id: str
    supervisor_pid: int
    supervisor_start_token: float
    graph_run_id: str | None
    node_id: int | None
    invocation_id: str
    pid: int
    pgid: int
    start_token: float
    helper_pid: int | None
    helper_start_token: float | None
    helper_generation: int
    ready_generation: int
    snapshot_seq: int
    observation_owner: str | None
    observation_seq: int | None
    descendants: tuple[Descendant, ...]
    started_at: str
    ended_at: str | None


def _record(row: sqlite3.Row) -> WorkerRecord:
    data = msgspec.json.decode(cast(str, row["descendants_json"]), type=list[Descendant])
    return WorkerRecord(
        runtime_run_id=cast(str, row["runtime_run_id"]),
        supervisor_pid=cast(int, row["supervisor_pid"]),
        supervisor_start_token=cast(float, row["supervisor_start_token"]),
        graph_run_id=cast(str | None, row["graph_run_id"]),
        node_id=cast(int | None, row["node_id"]),
        invocation_id=cast(str, row["invocation_id"]),
        pid=cast(int, row["pid"]),
        pgid=cast(int, row["pgid"]),
        start_token=cast(float, row["start_token"]),
        helper_pid=cast(int | None, row["helper_pid"]),
        helper_start_token=cast(float | None, row["helper_start_token"]),
        helper_generation=cast(int, row["helper_generation"]),
        ready_generation=cast(int, row["ready_generation"]),
        snapshot_seq=cast(int, row["snapshot_seq"]),
        observation_owner=cast(str | None, row["observation_owner"]),
        observation_seq=cast(int | None, row["observation_seq"]),
        descendants=tuple(data),
        started_at=cast(str, row["started_at"]),
        ended_at=cast(str | None, row["ended_at"]),
    )


def record_worker(conn: sqlite3.Connection, owner: WorkerOwner, worker: WorkerIdentity) -> None:
    if (owner.graph_run_id is None) != (owner.node_id is None):
        raise ValueError("graph run and node association must be set together")
    if not owner.runtime_run_id or owner.supervisor_pid <= 0 or owner.supervisor_start_token <= 0:
        raise ValueError("invalid worker owner")
    now = datetime.now(UTC).isoformat()
    columns = (
        "INSERT INTO run_workers "
        "(runtime_run_id, supervisor_pid, supervisor_start_token, graph_run_id, node_id, "
        "invocation_id, pid, pgid, start_token, started_at) "
    )
    values = (
        owner.runtime_run_id, owner.supervisor_pid, owner.supervisor_start_token,
        worker.invocation_id, worker.pid, worker.pgid, worker.start_token, now,
    )
    if owner.graph_run_id is None:
        cur = conn.execute(
            columns + "VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)", values,
        )
    else:
        cur = conn.execute(
            columns + "SELECT ?, ?, ?, run_id, node_id, ?, ?, ?, ?, ? FROM runs "
            "WHERE run_id = ? AND node_id = ? AND status = 'running'",
            (*values, owner.graph_run_id, owner.node_id),
        )
    conn.commit()
    if cur.rowcount != 1:
        raise RuntimeError(f"running run not found: {owner.graph_run_id}")


def live_workers(
    conn: sqlite3.Connection, *, node_id: int | None = None,
    run_id: str | None = None, unassociated: bool = False,
) -> tuple[WorkerRecord, ...]:
    rows = fetchall(
        conn,
        "SELECT * FROM run_workers WHERE ended_at IS NULL "
        "AND (? IS NULL OR node_id = ?) AND (? IS NULL OR graph_run_id = ?) "
        "AND (? = 0 OR graph_run_id IS NULL) "
        "ORDER BY started_at, invocation_id",
        (node_id, node_id, run_id, run_id, int(unassociated)),
    )
    return tuple(_record(cast(sqlite3.Row, row)) for row in rows)


def get_worker(conn: sqlite3.Connection, invocation_id: str) -> WorkerRecord | None:
    row = fetchone(conn, "SELECT * FROM run_workers WHERE invocation_id = ?", (invocation_id,))
    return None if row is None else _record(cast(sqlite3.Row, row))


def record_helper(conn: sqlite3.Connection, helper: HelperIdentity) -> None:
    cur = conn.execute(
        "UPDATE run_workers SET helper_pid = ?, helper_start_token = ?, "
        "helper_generation = ?, ready_generation = -1 "
        "WHERE invocation_id = ? AND ended_at IS NULL AND helper_generation = ?",
        (
            helper.pid,
            helper.start_token,
            helper.generation,
            helper.invocation_id,
            helper.generation - 1,
        ),
    )
    conn.commit()
    if cur.rowcount != 1:
        raise RuntimeError("stale helper generation")


def begin_observation(conn: sqlite3.Connection, key: ObservationKey) -> None:
    if key.owner not in ("supervisor", "helper"):
        raise ValueError("invalid worker observer")
    cur = conn.execute(
        "UPDATE run_workers SET observation_owner = ?, observation_seq = ?, "
        "observation_generation = ?, observation_pid = ?, observation_token = ? "
        "WHERE invocation_id = ? AND ended_at IS NULL AND observation_owner IS NULL "
        "AND snapshot_seq + 1 = ? AND "
        "(? = 'supervisor' OR (helper_generation = ? AND helper_pid = ? "
        "AND helper_start_token = ?))",
        (
            key.owner, key.sequence, key.generation, key.pid, key.start_token,
            key.invocation_id, key.sequence, key.owner, key.generation,
            key.pid, key.start_token,
        ),
    )
    conn.commit()
    if cur.rowcount != 1:
        raise RuntimeError("worker observation conflict")


def commit_observation(
    conn: sqlite3.Connection, key: ObservationKey, descendants: tuple[Descendant, ...]
) -> None:
    row = fetchone(
        conn,
        "SELECT descendants_json FROM run_workers WHERE invocation_id = ? "
        "AND ended_at IS NULL AND observation_owner = ? AND observation_seq = ? "
        "AND observation_generation = ? AND observation_pid = ? AND observation_token = ? "
        "AND (? = 'supervisor' OR (helper_generation = ? AND helper_pid = ? "
        "AND helper_start_token = ?))",
        (
            key.invocation_id, key.owner, key.sequence, key.generation,
            key.pid, key.start_token, key.owner, key.generation, key.pid, key.start_token,
        ),
    )
    if row is None:
        raise RuntimeError("worker observation fence lost")
    retained = msgspec.json.decode(cast(str, row[0]), type=list[Descendant])
    merged = sorted(set(retained) | set(descendants))
    cur = conn.execute(
        "UPDATE run_workers SET descendants_json = ?, snapshot_seq = ?, "
        "observation_owner = NULL, observation_seq = NULL, observation_generation = NULL, "
        "observation_pid = NULL, observation_token = NULL "
        "WHERE invocation_id = ? AND observation_owner = ? AND observation_seq = ? "
        "AND observation_generation = ? AND observation_pid = ? AND observation_token = ? "
        "AND (? = 'supervisor' OR (helper_generation = ? AND helper_pid = ? "
        "AND helper_start_token = ?))",
        (
            msgspec.json.encode(merged).decode(), key.sequence, key.invocation_id,
            key.owner, key.sequence, key.generation, key.pid, key.start_token,
            key.owner, key.generation, key.pid, key.start_token,
        ),
    )
    conn.commit()
    if cur.rowcount != 1:
        raise RuntimeError("worker observation fence lost")


def ready_helper(conn: sqlite3.Connection, helper: HelperIdentity, sequence: int) -> bool:
    cur = conn.execute(
        "UPDATE run_workers SET ready_generation = ? "
        "WHERE invocation_id = ? AND ended_at IS NULL "
        "AND helper_generation = ? AND helper_pid = ? AND helper_start_token = ? "
        "AND snapshot_seq = ? AND observation_owner IS NULL",
        (
            helper.generation, helper.invocation_id, helper.generation,
            helper.pid, helper.start_token, sequence,
        ),
    )
    conn.commit()
    return cur.rowcount == 1


def end_worker(
    conn: sqlite3.Connection, invocation_id: str, snapshot_seq: int,
    helper_generation: int | None = None,
) -> None:
    cur = conn.execute(
        "UPDATE run_workers SET ended_at = ? WHERE invocation_id = ? "
        "AND snapshot_seq = ? AND ended_at IS NULL AND observation_owner IS NULL "
        "AND (? IS NULL OR helper_generation = ?)",
        (datetime.now(UTC).isoformat(), invocation_id, snapshot_seq,
         helper_generation, helper_generation),
    )
    conn.commit()
    if cur.rowcount != 1:
        raise RuntimeError("worker snapshot, helper, or observation fence lost")
