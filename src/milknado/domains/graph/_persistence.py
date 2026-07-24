"""DB schema creation, row serialization, and admin helpers for MikadoGraph."""

from __future__ import annotations

import itertools
import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus, RunResult

if TYPE_CHECKING:
    from milknado.domains.batching import BatchPlan

_logger = logging.getLogger(__name__)


_REQUIRED_NODE_COLUMNS = frozenset(
    {
        "id",
        "description",
        "status",
        "parent_id",
        "worktree_path",
        "branch_name",
        "run_id",
        "pid",
        "created_at",
        "completed_at",
        "dispatched_at",
        "completion_duration_seconds",
        "oversized",
        "batch_index",
        "kind",
        "flavor",
        "wiki_ref",
        "artifact_path",
        "github_ref",
        "archived_at",
    }
)


# Ordered (user_version, single-statement SQL) migrations, applied by migrate().
# v1 is the create_tables schema; entries start at 2. Each statement must be a
# single SQL statement — migrate() executes them inside one transaction, and
# executescript() would COMMIT behind its back.
MIGRATIONS: list[tuple[int, str]] = [
    (
        2,
        "CREATE TABLE IF NOT EXISTS node_reviews ("
        "node_id INTEGER NOT NULL REFERENCES nodes(id), "
        "round INTEGER NOT NULL, "
        "verdict TEXT NOT NULL, "
        "findings TEXT NOT NULL, "
        "created_at TEXT NOT NULL, "
        "PRIMARY KEY (node_id, round))",
    ),
    (3, "ALTER TABLE nodes ADD COLUMN archived_at TEXT"),
]

SCHEMA_VERSION = max(version for version, _ in MIGRATIONS)

# Fresh databases are created with the full current schema at user_version=0,
# so migrate() always runs against them: an ALTER ... ADD COLUMN whose column
# is already present must be skipped (still stamping user_version) rather than
# failing with "duplicate column name".
_ADD_COLUMN_RE = re.compile(r"ALTER TABLE (\w+) ADD COLUMN (\w+)", re.IGNORECASE)


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})"))  # noqa: S608


def migrate(conn: sqlite3.Connection) -> None:
    """Apply pending MIGRATIONS in version order inside one transaction.

    Stamps PRAGMA user_version after each version. Any failure rolls the whole
    batch back and re-raises — a half-migrated database must never be stamped.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    pending = [(v, sql) for v, sql in MIGRATIONS if v > current]
    if not pending:
        return
    try:
        conn.execute("BEGIN")
        for version, sql in pending:
            add_column = _ADD_COLUMN_RE.match(sql.strip())
            if add_column is None or not _has_column(conn, *add_column.groups()):
                conn.execute(sql)
            conn.execute(f"PRAGMA user_version = {version}")  # noqa: S608
        conn.commit()
    except Exception:
        conn.rollback()
        _logger.exception(
            "migration failed at user_version=%d; rolled back pending %s",
            current,
            [v for v, _ in pending],
        )
        raise
    _logger.info("migrated database user_version %d -> %d", current, pending[-1][0])


def _validate_schema(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()}
    missing = sorted(_REQUIRED_NODE_COLUMNS - columns)
    if missing:
        raise RuntimeError(
            "obsolete milknado database schema; recreate the database "
            f"(missing nodes columns: {', '.join(missing)})"
        )


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_wiki_ref
            ON nodes(wiki_ref) WHERE wiki_ref IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_github_ref
            ON nodes(github_ref) WHERE github_ref IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_nodes_ready
            ON nodes(status, kind, flavor, id);
        CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_id);
        CREATE INDEX IF NOT EXISTS idx_file_ownership_path
            ON file_ownership(file_path, node_id);
        CREATE INDEX IF NOT EXISTS idx_run_messages_latest_role
            ON run_messages(run_id, role, seq DESC);
        CREATE INDEX IF NOT EXISTS idx_run_messages_result_latest
            ON run_messages(role, run_id, created_at DESC, seq DESC, body);
        CREATE TABLE IF NOT EXISTS github_bind_attempts (
            goal_id INTEGER PRIMARY KEY REFERENCES nodes(id),
            marker TEXT NOT NULL,
            issue_url TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_github_bind_attempts_marker
            ON github_bind_attempts(marker);
        """
    )
    conn.commit()


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the current schema or validate an existing database."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'nodes'"
    ).fetchone()
    if exists is not None:
        # Migrate BEFORE validating: a migration may add a nodes column that
        # _REQUIRED_NODE_COLUMNS already demands, so validating first would
        # condemn a database that one pending migration would heal.
        migrate(conn)
        _validate_schema(conn)
        _ensure_indexes(conn)
        return
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
            flavor TEXT,
            wiki_ref TEXT,
            artifact_path TEXT,
            github_ref TEXT,
            archived_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_wiki_ref
            ON nodes(wiki_ref) WHERE wiki_ref IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_github_ref
            ON nodes(github_ref) WHERE github_ref IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_nodes_ready
            ON nodes(status, kind, flavor, id);
        CREATE TABLE IF NOT EXISTS edges (
            parent_id INTEGER NOT NULL,
            child_id INTEGER NOT NULL,
            PRIMARY KEY (parent_id, child_id),
            FOREIGN KEY (parent_id) REFERENCES nodes(id),
            FOREIGN KEY (child_id) REFERENCES nodes(id)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_id);
        CREATE TABLE IF NOT EXISTS file_ownership (
            node_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            PRIMARY KEY (node_id, file_path),
            FOREIGN KEY (node_id) REFERENCES nodes(id)
        );
        CREATE INDEX IF NOT EXISTS idx_file_ownership_path
            ON file_ownership(file_path, node_id);
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
        CREATE INDEX IF NOT EXISTS idx_run_messages_latest_role
            ON run_messages(run_id, role, seq DESC);
        CREATE INDEX IF NOT EXISTS idx_run_messages_result_latest
            ON run_messages(role, run_id, created_at DESC, seq DESC, body);
        CREATE TABLE IF NOT EXISTS goal_claims (
            goal_id     INTEGER PRIMARY KEY REFERENCES nodes(id),
            run_id      TEXT NOT NULL,
            pid         INTEGER,
            claimed_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS github_bind_attempts (
            goal_id INTEGER PRIMARY KEY REFERENCES nodes(id),
            marker TEXT NOT NULL,
            issue_url TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_github_bind_attempts_marker
            ON github_bind_attempts(marker);
    """)
    conn.commit()


def row_to_node(row: sqlite3.Row) -> MikadoNode:
    kind = NodeKind(row["kind"])
    flavor = row["flavor"]
    if kind != NodeKind.TASK and flavor is not None:
        raise ValueError(
            f"node {row['id']} has kind={kind.value} but a non-NULL flavor={flavor!r}; "
            "only task nodes may carry a flavor"
        )
    completed_at = row["completed_at"]
    dispatched_at = row["dispatched_at"]
    archived_at = row["archived_at"]
    return MikadoNode(
        id=row["id"],
        description=row["description"],
        status=NodeStatus(row["status"]),
        parent_id=row["parent_id"],
        worktree_path=row["worktree_path"],
        branch_name=row["branch_name"],
        run_id=row["run_id"],
        pid=row["pid"],
        created_at=datetime.fromisoformat(row["created_at"]),
        completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
        dispatched_at=datetime.fromisoformat(dispatched_at) if dispatched_at else None,
        archived_at=datetime.fromisoformat(archived_at) if archived_at else None,
        oversized=bool(row["oversized"]),
        batch_index=row["batch_index"],
        completion_duration_seconds=row["completion_duration_seconds"],
        kind=kind,
        flavor=flavor if kind == NodeKind.TASK else None,
        wiki_ref=row["wiki_ref"],
        github_ref=row["github_ref"],
        artifact_path=row["artifact_path"],
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


def get_file_ownership_map(
    conn: sqlite3.Connection, node_ids: Iterable[int] | None = None
) -> dict[int, list[str]]:
    """Return owned paths for the requested nodes from one candidate-set scan."""
    if node_ids is None:
        rows = conn.execute(
            "SELECT node_id, file_path FROM file_ownership ORDER BY node_id, file_path"
        ).fetchall()
    else:
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT node_id, file_path FROM file_ownership "  # noqa: S608
            f"WHERE node_id IN ({placeholders}) ORDER BY node_id, file_path",
            ids,
        ).fetchall()
    mapping: dict[int, list[str]] = {}
    for node_id, file_path in rows:
        mapping.setdefault(node_id, []).append(file_path)
    return mapping


def get_github_bind_attempt(conn: sqlite3.Connection, goal_id: int) -> dict | None:
    row = conn.execute(
        "SELECT goal_id, marker, issue_url, created_at "
        "FROM github_bind_attempts WHERE goal_id = ?",
        (goal_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def set_github_bind_attempt(
    conn: sqlite3.Connection,
    goal_id: int,
    marker: str,
    issue_url: str | None,
    created_at: str,
) -> None:
    conn.execute(
        "INSERT INTO github_bind_attempts (goal_id, marker, issue_url, created_at) "
        "VALUES (?, ?, ?, ?) ON CONFLICT(goal_id) DO UPDATE SET "
        "marker = excluded.marker, issue_url = excluded.issue_url, "
        "created_at = excluded.created_at",
        (goal_id, marker, issue_url, created_at),
    )
    conn.commit()


def clear_github_bind_attempt(conn: sqlite3.Connection, goal_id: int) -> None:
    conn.execute("DELETE FROM github_bind_attempts WHERE goal_id = ?", (goal_id,))
    conn.commit()


def check_parallel_safety(
    conn: sqlite3.Connection, node_ids: list[int]
) -> list[tuple[int, int, list[str]]]:
    if not node_ids:
        return []
    placeholders = ",".join("?" for _ in node_ids)
    rows = conn.execute(
        f"SELECT node_id, file_path FROM file_ownership "  # noqa: S608
        f"WHERE node_id IN ({placeholders})",
        node_ids,
    ).fetchall()
    ownership = {node_id: set() for node_id in node_ids}
    for node_id, file_path in rows:
        ownership[node_id].add(file_path)
    conflicts = []
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
    conn.execute("DELETE FROM goal_claims")
    conn.execute("DELETE FROM node_reviews")
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


def finish_run(conn: sqlite3.Connection, run_id: str, result: RunResult) -> bool:
    """Write one terminal result; return whether this fenced write won."""
    cur = conn.execute(
        "UPDATE runs SET status = ?, exit_code = ?, timed_out = ?, ended_at = ?, "
        "error = ?, detail = ?, rebased = ? WHERE run_id = ? AND status = 'running'",
        (
            result.status,
            result.exit_code,
            1 if result.timed_out else 0,
            result.ended_at,
            result.error,
            result.detail,
            None if result.rebased is None else (1 if result.rebased else 0),
            run_id,
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


def insert_node_review(
    conn: sqlite3.Connection,
    node_id: int,
    round_number: int,
    verdict: str,
    findings: str,
    created_at: str,
) -> None:
    """Persist one review verdict keyed by (node_id, round); no dependency on runs."""
    conn.execute(
        "INSERT INTO node_reviews (node_id, round, verdict, findings, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (node_id, round_number, verdict, findings, created_at),
    )
    conn.commit()


def node_reviews_for_node(conn: sqlite3.Connection, node_id: int) -> list[dict]:
    """Return all recorded review verdicts for a node, round order."""
    rows = conn.execute(
        "SELECT node_id, round, verdict, findings, created_at FROM node_reviews "
        "WHERE node_id = ? ORDER BY round",
        (node_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def latest_run_message(conn: sqlite3.Connection, run_id: str, role: str) -> str | None:
    """Return the body of the latest message for this run+role, or None."""
    row = conn.execute(
        "SELECT body FROM run_messages WHERE run_id = ? AND role = ? ORDER BY seq DESC LIMIT 1",
        (run_id, role),
    ).fetchone()
    return row[0] if row else None
