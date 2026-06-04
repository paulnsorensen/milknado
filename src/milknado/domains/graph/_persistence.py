"""DB schema creation, row serialization, and admin helpers for MikadoGraph."""

from __future__ import annotations

import itertools
import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus
from milknado.domains.common.types import TaskFlavor

if TYPE_CHECKING:
    from milknado.domains.batching import BatchPlan

_logger = logging.getLogger(__name__)


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            parent_id INTEGER,
            worktree_path TEXT,
            branch_name TEXT,
            run_id TEXT,
            pid INTEGER,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            dispatched_at TEXT,
            completion_duration_seconds REAL,
            oversized INTEGER NOT NULL DEFAULT 0,
            batch_index INTEGER,
            kind TEXT NOT NULL DEFAULT 'task',
            flavor TEXT
        );
        CREATE TABLE IF NOT EXISTS edges (
            parent_id INTEGER NOT NULL,
            child_id INTEGER NOT NULL,
            PRIMARY KEY (parent_id, child_id),
            FOREIGN KEY (parent_id) REFERENCES nodes(id),
            FOREIGN KEY (child_id) REFERENCES nodes(id)
        );
        CREATE TABLE IF NOT EXISTS file_ownership (
            node_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            PRIMARY KEY (node_id, file_path),
            FOREIGN KEY (node_id) REFERENCES nodes(id)
        );
        CREATE TABLE IF NOT EXISTS plan_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            spec_hash TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS batch_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            solver_status TEXT NOT NULL,
            batch_count INTEGER NOT NULL,
            oversized_count INTEGER NOT NULL,
            max_spread INTEGER NOT NULL,
            spread_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id          TEXT PRIMARY KEY,
            node_id         INTEGER NOT NULL REFERENCES nodes(id),
            status          TEXT NOT NULL,
            pid             INTEGER,
            log_path        TEXT NOT NULL,
            started_at      TEXT NOT NULL,
            ended_at        TEXT,
            timed_out       INTEGER NOT NULL DEFAULT 0,
            exit_code       INTEGER,
            error           TEXT,
            timeout_seconds INTEGER,
            detail          TEXT,
            rebased         INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_runs_node_status ON runs(node_id, status);
        CREATE TABLE IF NOT EXISTS run_messages (
            run_id     TEXT NOT NULL REFERENCES runs(run_id),
            seq        INTEGER NOT NULL,
            role       TEXT NOT NULL,
            body       TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, seq)
        );
        CREATE TABLE IF NOT EXISTS goal_claims (
            goal_id     INTEGER PRIMARY KEY REFERENCES nodes(id),
            run_id      TEXT NOT NULL,
            pid         INTEGER,
            claimed_at  TEXT NOT NULL
        );
    """)


def ensure_schema(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()}
    for col, ddl in [
        ("run_id", "ALTER TABLE nodes ADD COLUMN run_id TEXT"),
        (
            "completion_duration_seconds",
            "ALTER TABLE nodes ADD COLUMN completion_duration_seconds REAL",
        ),
        ("dispatched_at", "ALTER TABLE nodes ADD COLUMN dispatched_at TEXT"),
        ("kind", "ALTER TABLE nodes ADD COLUMN kind TEXT NOT NULL DEFAULT 'task'"),
        ("pid", "ALTER TABLE nodes ADD COLUMN pid INTEGER"),
        ("flavor", "ALTER TABLE nodes ADD COLUMN flavor TEXT"),
    ]:
        if col not in columns:
            conn.execute(ddl)
            conn.commit()


def row_to_node(row: sqlite3.Row) -> MikadoNode:
    keys = row.keys()
    run_id = row["run_id"] if "run_id" in keys else None
    pid = row["pid"] if "pid" in keys else None
    oversized = bool(row["oversized"]) if "oversized" in keys else False
    batch_index = row["batch_index"] if "batch_index" in keys else None
    dispatched_at_raw = row["dispatched_at"] if "dispatched_at" in keys else None
    col = "completion_duration_seconds"
    duration = row[col] if col in keys else None
    completed_at_raw = row["completed_at"]
    kind = NodeKind(row["kind"]) if "kind" in keys else NodeKind.TASK
    flavor_raw = row["flavor"] if "flavor" in keys else None
    if kind == NodeKind.TASK:
        flavor = TaskFlavor(flavor_raw) if flavor_raw is not None else None
    elif flavor_raw is not None:
        raise ValueError(
            f"node {row['id']} has kind={kind.value} but a non-NULL flavor={flavor_raw!r}; "
            "only task nodes may carry a flavor"
        )
    else:
        flavor = None
    return MikadoNode(
        id=row["id"],
        description=row["description"],
        status=NodeStatus(row["status"]),
        parent_id=row["parent_id"],
        worktree_path=row["worktree_path"],
        branch_name=row["branch_name"],
        run_id=run_id,
        pid=pid,
        created_at=datetime.fromisoformat(row["created_at"]),
        completed_at=(datetime.fromisoformat(completed_at_raw) if completed_at_raw else None),
        dispatched_at=(datetime.fromisoformat(dispatched_at_raw) if dispatched_at_raw else None),
        oversized=oversized,
        batch_index=batch_index,
        completion_duration_seconds=duration,
        kind=kind,
        flavor=flavor,
    )


def children_id_map(conn: sqlite3.Connection) -> dict[int, list[int]]:
    """parent_id -> child_ids from a single edges scan (avoids per-node queries)."""
    mapping: dict[int, list[int]] = {}
    for parent_id, child_id in conn.execute("SELECT parent_id, child_id FROM edges").fetchall():
        mapping.setdefault(parent_id, []).append(child_id)
    return mapping


def set_file_ownership(conn: sqlite3.Connection, node_id: int, files: list[str]) -> None:
    conn.execute("DELETE FROM file_ownership WHERE node_id = ?", (node_id,))
    conn.executemany(
        "INSERT INTO file_ownership (node_id, file_path) VALUES (?, ?)",
        [(node_id, f) for f in files],
    )
    conn.commit()


def get_file_ownership(conn: sqlite3.Connection, node_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT file_path FROM file_ownership WHERE node_id = ?", (node_id,)
    ).fetchall()
    return [r[0] for r in rows]


def check_parallel_safety(
    conn: sqlite3.Connection, node_ids: list[int]
) -> list[tuple[int, int, list[str]]]:
    ownership: dict[int, set[str]] = {nid: set(get_file_ownership(conn, nid)) for nid in node_ids}
    conflicts: list[tuple[int, int, list[str]]] = []
    for left_id, right_id in itertools.combinations(node_ids, 2):
        overlap = ownership[left_id] & ownership[right_id]
        if overlap:
            conflicts.append((left_id, right_id, sorted(overlap)))
    return conflicts


def record_batch_plan(conn: sqlite3.Connection, plan: BatchPlan) -> int:
    spread_payload = [
        {"symbol_name": item.symbol.name, "symbol_file": item.symbol.file, "spread": item.spread}
        for item in plan.spread_report
    ]
    max_spread = max((item.spread for item in plan.spread_report), default=0)
    oversized_count = sum(1 for b in plan.batches if b.oversized)
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        "INSERT INTO batch_plans "
        "(created_at, solver_status, batch_count, oversized_count, max_spread, spread_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            now,
            plan.solver_status,
            len(plan.batches),
            oversized_count,
            max_spread,
            json.dumps(spread_payload),
        ),
    )
    conn.commit()
    plan_id = cur.lastrowid
    if plan_id is None:  # pragma: no cover - defensive: plain INSERT always sets lastrowid
        raise RuntimeError("record_batch_plan INSERT did not return lastrowid")
    return plan_id


def get_latest_batch_plan(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        "SELECT id, created_at, solver_status, batch_count, oversized_count, "
        "max_spread, spread_json FROM batch_plans ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "solver_status": row["solver_status"],
        "batch_count": row["batch_count"],
        "oversized_count": row["oversized_count"],
        "max_spread": row["max_spread"],
        "spread_report": json.loads(row["spread_json"]),
    }


def set_spec_hash(conn: sqlite3.Connection, spec_hash: str) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO plan_state (id, spec_hash, recorded_at) VALUES (1, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET spec_hash = excluded.spec_hash,"
        " recorded_at = excluded.recorded_at",
        (spec_hash, now),
    )
    conn.commit()


def get_spec_hash(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT spec_hash FROM plan_state WHERE id = 1").fetchone()
    return row["spec_hash"] if row else None


def drop_all(conn: sqlite3.Connection) -> int:
    count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    conn.execute("DELETE FROM run_messages")
    conn.execute("DELETE FROM runs")
    conn.execute("DELETE FROM file_ownership")
    conn.execute("DELETE FROM edges")
    conn.execute("DELETE FROM nodes")
    conn.execute("DELETE FROM plan_state")
    conn.execute("DELETE FROM batch_plans")
    conn.commit()
    return count


def record_completion_duration(conn: sqlite3.Connection, node_id: int, duration: float) -> None:
    conn.execute(
        "UPDATE nodes SET completion_duration_seconds = ? WHERE id = ?", (duration, node_id)
    )
    conn.commit()


def recent_completion_durations(conn: sqlite3.Connection, limit: int) -> list[float]:
    rows = conn.execute(
        "SELECT completion_duration_seconds FROM nodes "
        "WHERE completion_duration_seconds IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r[0] for r in rows]


def set_dispatched_at(conn: sqlite3.Connection, node_id: int) -> None:
    cur = conn.execute(
        "UPDATE nodes SET dispatched_at = ? WHERE id = ?",
        (datetime.now(UTC).isoformat(), node_id),
    )
    if cur.rowcount == 0:
        raise ValueError(f"Node {node_id} not found")
    conn.commit()


def set_pid(conn: sqlite3.Connection, node_id: int, run_id: str, pid: int) -> None:
    """Record the worker pid on the node, gated on the fence (current run_id).

    A CAS on the owning run_id: if the node has since been reclaimed under a new
    run_id, the write hits zero rows and is silently dropped — the pid belongs to
    a run that no longer owns the node, so recording it would be wrong.
    """
    conn.execute(
        "UPDATE nodes SET pid = ? WHERE id = ? AND run_id = ?",
        (pid, node_id, run_id),
    )
    conn.commit()


def set_worktree(
    conn: sqlite3.Connection, node_id: int, run_id: str, worktree_path: str, branch_name: str
) -> None:
    """Attach worktree/branch to a node without a status transition, gated on the
    fence (current run_id). Used by the executor when the node was already claimed
    RUNNING by the dispatching parent: re-marking RUNNING would be an illegal
    RUNNING -> RUNNING transition, so only the worktree metadata is written, and
    only if this run still owns the node.
    """
    conn.execute(
        "UPDATE nodes SET worktree_path = ?, branch_name = ? WHERE id = ? AND run_id = ?",
        (worktree_path, branch_name, node_id, run_id),
    )
    conn.commit()


_RUN_COLUMNS = (
    "run_id",
    "node_id",
    "status",
    "pid",
    "log_path",
    "started_at",
    "ended_at",
    "timed_out",
    "exit_code",
    "error",
    "timeout_seconds",
    "detail",
    "rebased",
)


def _run_row_to_dict(row: sqlite3.Row) -> dict:
    """Serialize a runs row to the sidecar-shaped dict callers consumed before.

    `timed_out` / `rebased` are stored as INTEGER and re-hydrated to bool/None so
    a poll payload reads the same as the old JSON sidecar (which carried bools).
    """
    d = {col: row[col] for col in _RUN_COLUMNS}
    d["timed_out"] = bool(d["timed_out"]) if d["timed_out"] is not None else False
    if d["rebased"] is not None:
        d["rebased"] = bool(d["rebased"])
    return d


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
    conn.execute(
        "INSERT INTO runs "
        "(run_id, node_id, status, pid, log_path, started_at, timeout_seconds, timed_out) "
        "VALUES (?, ?, 'running', ?, ?, ?, ?, 0)",
        (run_id, node_id, pid, log_path, started_at, timeout_seconds),
    )
    conn.commit()


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str,
    exit_code: int | None,
    timed_out: bool,
    ended_at: str,
    error: str | None = None,
    detail: str | None = None,
    rebased: bool | None = None,
) -> None:
    """UPDATE a running run to a terminal status; first terminal write wins.

    Gated on status = 'running' so a late terminal write (e.g. a wedged worker
    finishing after milknado_run_cancel already finalized the run) cannot
    clobber an already-terminal row back to a different outcome.
    """
    cur = conn.execute(
        "UPDATE runs SET status = ?, exit_code = ?, timed_out = ?, ended_at = ?, "
        "error = ?, detail = ?, rebased = ? WHERE run_id = ? AND status = 'running'",
        (
            status,
            exit_code,
            1 if timed_out else 0,
            ended_at,
            error,
            detail,
            None if rebased is None else (1 if rebased else 0),
            run_id,
        ),
    )
    conn.commit()
    if cur.rowcount == 0:
        _logger.warning(
            "finish_run dropped late terminal write for run %s (status=%s)", run_id, status
        )


def set_run_pid(conn: sqlite3.Connection, run_id: str, pid: int) -> None:
    """Record the detached runner's pid on a still-running run.

    Gated on status='running' so a runner that already wrote its terminal state
    is never clobbered back toward running (mirrors the old read-then-write
    guard the sidecar path used).
    """
    conn.execute(
        "UPDATE runs SET pid = ? WHERE run_id = ? AND status = 'running'",
        (pid, run_id),
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    """SELECT one run row as a dict (replaces read_state for poll)."""
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return _run_row_to_dict(row) if row else None


def runs_for_node(
    conn: sqlite3.Connection,
    node_id: int,
    *,
    terminal_only: bool = False,
    run_id: str | None = None,
) -> list[dict]:
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
    rows = conn.execute(sql, params).fetchall()
    return [_run_row_to_dict(r) for r in rows]


def recent_runs(conn: sqlite3.Connection, limit: int) -> list[dict]:
    """Return the most-recently-started runs as dicts (replaces the run-list scan)."""
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (max(limit, 0),)
    ).fetchall()
    return [_run_row_to_dict(r) for r in rows]


def deposit_run_message(
    conn: sqlite3.Connection, run_id: str, role: str, body: str, created_at: str
) -> int:
    """Append a run_messages row with the next seq for this run; return that seq.

    Seq assignment and insert happen in one statement so concurrent depositors
    for the same run cannot race MAX(seq) into a UNIQUE collision.
    """
    row = conn.execute(
        "INSERT INTO run_messages (run_id, seq, role, body, created_at) "
        "SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ? FROM run_messages WHERE run_id = ? "
        "RETURNING seq",
        (run_id, role, body, created_at, run_id),
    ).fetchone()
    conn.commit()
    return row[0]


def latest_run_message(conn: sqlite3.Connection, run_id: str, role: str) -> str | None:
    """Return the body of the latest message for this run+role, or None."""
    row = conn.execute(
        "SELECT body FROM run_messages WHERE run_id = ? AND role = ? ORDER BY seq DESC LIMIT 1",
        (run_id, role),
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Goal-claim persistence (coordinator subtree fencing)
# ---------------------------------------------------------------------------


def claim_goal_row(
    conn: sqlite3.Connection, goal_id: int, run_id: str, now: str, *, pid: int | None = None
) -> bool:
    """INSERT a goal claim if none exists; returns True iff this caller won.

    INSERT OR IGNORE is the mutual-exclusion point: a single conditional write
    that either inserts (this caller wins) or finds the row already there (loses).
    Pass pid to write it atomically in the same INSERT, eliminating the NULL-pid
    window that would otherwise allow a concurrent reclaim before set_goal_claim_pid.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
        (goal_id, run_id, pid, now),
    )
    conn.commit()
    return cur.rowcount == 1


def release_goal_row(conn: sqlite3.Connection, goal_id: int, run_id: str) -> bool:
    """Delete a goal claim gated on the owning run_id. Returns True iff deleted."""
    cur = conn.execute(
        "DELETE FROM goal_claims WHERE goal_id = ? AND run_id = ?",
        (goal_id, run_id),
    )
    conn.commit()
    return cur.rowcount == 1


def release_goal_row_unconditional(conn: sqlite3.Connection, goal_id: int) -> None:
    """Delete any goal claim for goal_id without a run_id gate.

    Used on terminal transitions (DONE/FAILED) for goal nodes: the goal is
    complete so the fence owner is moot — the claim must be cleared regardless
    of which run held it.
    """
    conn.execute("DELETE FROM goal_claims WHERE goal_id = ?", (goal_id,))
    conn.commit()


def set_goal_claim_pid(conn: sqlite3.Connection, goal_id: int, run_id: str, pid: int) -> None:
    """Record the coordinator pid on a goal claim, gated on the owning run_id."""
    conn.execute(
        "UPDATE goal_claims SET pid = ? WHERE goal_id = ? AND run_id = ?",
        (pid, goal_id, run_id),
    )
    conn.commit()


def get_goal_claim(conn: sqlite3.Connection, goal_id: int) -> dict | None:
    """Return the goal claim row as a dict, or None if unclaimed."""
    row = conn.execute(
        "SELECT goal_id, run_id, pid, claimed_at FROM goal_claims WHERE goal_id = ?",
        (goal_id,),
    ).fetchone()
    if row is None:
        return None
    return {"goal_id": row[0], "run_id": row[1], "pid": row[2], "claimed_at": row[3]}


def ancestor_goal_claim(
    conn: sqlite3.Connection, node_id: int, caller_run_id: str | None
) -> dict | None:
    """Walk node's ancestor chain; return the first goal_claim owned by a DIFFERENT run.

    Returns None (allowed) when:
    - no ancestor goal exists
    - the ancestor goal is unclaimed
    - the claim belongs to the same caller_run_id (own coordinator)
    Returns the claim dict (blocked) when a live OR dead-but-not-yet-reclaimed foreign
    claim is found; the caller is responsible for checking pid-liveness.
    """
    # Walk parent chain collecting each ancestor's kind + claim status.
    current_id: int | None = node_id
    while current_id is not None:
        row = conn.execute(
            "SELECT parent_id, kind FROM nodes WHERE id = ?", (current_id,)
        ).fetchone()
        if row is None:
            break
        parent_id, kind = row["parent_id"], row["kind"]
        if kind == "goal":
            claim = get_goal_claim(conn, current_id)
            if claim is not None and claim["run_id"] != caller_run_id:
                return claim
        current_id = parent_id
    return None


def find_ancestor_goal_id(conn: sqlite3.Connection, node_id: int) -> int | None:
    """Return the id of the closest ancestor (or self) with kind='goal', or None."""
    current_id: int | None = node_id
    while current_id is not None:
        row = conn.execute(
            "SELECT parent_id, kind FROM nodes WHERE id = ?", (current_id,)
        ).fetchone()
        if row is None:
            break
        if row["kind"] == "goal":
            return current_id
        current_id = row["parent_id"]
    return None
