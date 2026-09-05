"""DB schema creation, row serialization, and admin helpers for MikadoGraph."""

from __future__ import annotations

import itertools
import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict, cast

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus
from milknado.domains.graph._run_persistence import (
    deposit_review_verdict,
    finish_run,
    get_run,
    insert_node_review,
    latest_run_message,
    node_reviews_for_node,
    recent_runs,
    runs_for_node,
    set_run_pid,
    start_run,
)
from milknado.domains.graph._sqlite_rows import as_tuple as _as_tuple
from milknado.domains.graph._sqlite_rows import fetchall, fetchone

__all__ = [
    "deposit_review_verdict",
    "finish_run",
    "get_run",
    "insert_node_review",
    "latest_run_message",
    "node_reviews_for_node",
    "recent_runs",
    "runs_for_node",
    "set_run_pid",
    "start_run",
]

if TYPE_CHECKING:
    from milknado.domains.batching import BatchPlan

_logger = logging.getLogger(__name__)


class _NodeRow(TypedDict):
    id: int
    description: str
    status: str
    parent_id: int | None
    worktree_path: str | None
    branch_name: str | None
    run_id: str | None
    pid: int | None
    created_at: str
    completed_at: str | None
    dispatched_at: str | None
    completion_duration_seconds: float | None  # noqa: V107 - TypedDict key, read via node[...] at row_to_node
    oversized: int
    batch_index: int | None
    kind: str
    flavor: str | None
    wiki_ref: str | None
    github_ref: str | None
    artifact_path: str | None
    archived_at: str | None


class GithubBindAttempt(TypedDict):
    goal_id: int
    marker: str
    issue_url: str | None
    created_at: str


class BatchPlanRecord(TypedDict):
    id: int
    created_at: str
    solver_status: str
    batch_count: int
    oversized_count: int
    max_spread: int
    spread_report: list[dict[str, int | str]]


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
        + "node_id INTEGER NOT NULL REFERENCES nodes(id), "
        + "round INTEGER NOT NULL, "
        + "verdict TEXT NOT NULL, "
        + "findings TEXT NOT NULL, "
        + "created_at TEXT NOT NULL, "
        + "PRIMARY KEY (node_id, round))",
    ),
    (3, "ALTER TABLE nodes ADD COLUMN archived_at TEXT"),
]

SCHEMA_VERSION = max(version for version, _ in MIGRATIONS)

# Fresh databases start at user_version=0, so migrate() always runs against them.
# create_tables() covers most of the current schema but NOT node_reviews — step v2
# is what creates that table, on old and fresh databases alike. Objects it does
# create (e.g. nodes.archived_at) make the matching step redundant, so an
# ALTER ... ADD COLUMN whose column is already present is skipped (still stamping
# user_version) rather than failing with "duplicate column name".
_ADD_COLUMN_RE = re.compile(r"ALTER TABLE (\w+) ADD COLUMN (\w+)", re.IGNORECASE)


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(
        cast(str, _as_tuple(row)[1]) == column
        for row in fetchall(conn, f"PRAGMA table_info({table})")
    )


def migrate(conn: sqlite3.Connection) -> None:
    """Apply pending MIGRATIONS in version order inside one transaction.

    Stamps PRAGMA user_version after each version. Any failure rolls the whole
    batch back and re-raises — a half-migrated database must never be stamped.
    """
    current_row = fetchone(conn, "PRAGMA user_version")
    if current_row is None:
        raise RuntimeError("PRAGMA user_version returned no row")
    current = cast(int, _as_tuple(current_row)[0])
    pending = [(v, sql) for v, sql in MIGRATIONS if v > current]
    if not pending:
        return
    try:
        _ = conn.execute("BEGIN")
        for version, sql in pending:
            add_column = _ADD_COLUMN_RE.match(sql.strip())
            if add_column is None or not _has_column(conn, *add_column.groups()):
                _ = conn.execute(sql)
            _ = conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
    except Exception:
        conn.rollback()
        _logger.exception(
            "migration failed at user_version=%d; rolled back pending %s",
            current,
            [v for v, _ in pending],
        )
        raise
    _logger.info("migrated database user_version %d -> %d", current, SCHEMA_VERSION)


def _validate_schema(conn: sqlite3.Connection) -> None:
    columns = {cast(str, _as_tuple(row)[1]) for row in fetchall(conn, "PRAGMA table_info(nodes)")}
    missing = sorted(_REQUIRED_NODE_COLUMNS - columns)
    if missing:
        raise RuntimeError(
            "obsolete milknado database schema; recreate the database "
            + f"(missing nodes columns: {', '.join(missing)})"
        )


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    _ = conn.executescript(
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
    exists = fetchone(conn, "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'nodes'")
    if exists is not None:
        # Migrate BEFORE validating: a migration may add a nodes column that
        # _REQUIRED_NODE_COLUMNS already demands, so validating first would
        # condemn a database that one pending migration would heal.
        migrate(conn)
        _validate_schema(conn)
        _ensure_indexes(conn)
        return
    _ = conn.executescript("""
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
    node = cast(_NodeRow, cast(object, row))
    kind = NodeKind(node["kind"])
    flavor = node["flavor"]
    if kind != NodeKind.TASK and flavor is not None:
        raise ValueError(
            f"node {node['id']} has kind={kind.value} but a non-NULL flavor={flavor!r}; "
            + "only task nodes may carry a flavor"
        )
    completed_at = node["completed_at"]
    dispatched_at = node["dispatched_at"]
    archived_at = node["archived_at"]
    return MikadoNode(
        id=node["id"],
        description=node["description"],
        status=NodeStatus(node["status"]),
        parent_id=node["parent_id"],
        worktree_path=node["worktree_path"],
        branch_name=node["branch_name"],
        run_id=node["run_id"],
        pid=node["pid"],
        created_at=datetime.fromisoformat(node["created_at"]),
        completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
        dispatched_at=datetime.fromisoformat(dispatched_at) if dispatched_at else None,
        archived_at=datetime.fromisoformat(archived_at) if archived_at else None,
        oversized=bool(node["oversized"]),
        batch_index=node["batch_index"],
        completion_duration_seconds=node["completion_duration_seconds"],
        kind=kind,
        flavor=flavor if kind == NodeKind.TASK else None,
        wiki_ref=node["wiki_ref"],
        github_ref=node["github_ref"],
        artifact_path=node["artifact_path"],
    )


def children_id_map(conn: sqlite3.Connection) -> dict[int, list[int]]:
    """parent_id -> child_ids from a single edges scan (avoids per-node queries)."""
    mapping: dict[int, list[int]] = {}
    for row in fetchall(conn, "SELECT parent_id, child_id FROM edges"):
        parent_id = cast(int, _as_tuple(row)[0])
        child_id = cast(int, _as_tuple(row)[1])
        mapping.setdefault(parent_id, []).append(child_id)
    return mapping


def set_file_ownership(conn: sqlite3.Connection, node_id: int, files: list[str]) -> None:
    _ = conn.execute("DELETE FROM file_ownership WHERE node_id = ?", (node_id,))
    _ = conn.executemany(
        "INSERT INTO file_ownership (node_id, file_path) VALUES (?, ?)",
        [(node_id, f) for f in files],
    )
    conn.commit()


def get_file_ownership(conn: sqlite3.Connection, node_id: int) -> list[str]:
    rows = fetchall(conn, "SELECT file_path FROM file_ownership WHERE node_id = ?", (node_id,))
    return [cast(str, _as_tuple(row)[0]) for row in rows]


def get_file_ownership_map(
    conn: sqlite3.Connection, node_ids: Iterable[int] | None = None
) -> dict[int, list[str]]:
    """Return owned paths for the requested nodes from one candidate-set scan."""
    if node_ids is None:
        rows = fetchall(
            conn, "SELECT node_id, file_path FROM file_ownership ORDER BY node_id, file_path"
        )
    else:
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = fetchall(
            conn,
            "SELECT node_id, file_path FROM file_ownership "
            + f"WHERE node_id IN ({placeholders}) ORDER BY node_id, file_path",
            ids,
        )
    mapping: dict[int, list[str]] = {}
    for row in rows:
        values = _as_tuple(row)
        node_id = cast(int, values[0])
        file_path = cast(str, values[1])
        mapping.setdefault(node_id, []).append(file_path)
    return mapping


def get_github_bind_attempt(conn: sqlite3.Connection, goal_id: int) -> GithubBindAttempt | None:
    row = fetchone(
        conn,
        "SELECT goal_id, marker, issue_url, created_at "
        + "FROM github_bind_attempts WHERE goal_id = ?",
        (goal_id,),
    )
    if row is None:
        return None
    values = _as_tuple(row)
    return {
        "goal_id": cast(int, values[0]),
        "marker": cast(str, values[1]),
        "issue_url": cast(str | None, values[2]),
        "created_at": cast(str, values[3]),
    }


def set_github_bind_attempt(
    conn: sqlite3.Connection,
    goal_id: int,
    marker: str,
    issue_url: str | None,
    created_at: str,
) -> None:
    _ = conn.execute(
        "INSERT INTO github_bind_attempts (goal_id, marker, issue_url, created_at) "
        + "VALUES (?, ?, ?, ?) ON CONFLICT(goal_id) DO UPDATE SET "
        + "marker = excluded.marker, issue_url = excluded.issue_url, "
        + "created_at = excluded.created_at",
        (goal_id, marker, issue_url, created_at),
    )
    conn.commit()


def clear_github_bind_attempt(conn: sqlite3.Connection, goal_id: int) -> None:
    _ = conn.execute("DELETE FROM github_bind_attempts WHERE goal_id = ?", (goal_id,))
    conn.commit()


def check_parallel_safety(
    conn: sqlite3.Connection, node_ids: list[int]
) -> list[tuple[int, int, list[str]]]:
    if not node_ids:
        return []
    placeholders = ",".join("?" for _ in node_ids)
    rows = fetchall(
        conn,
        f"SELECT node_id, file_path FROM file_ownership WHERE node_id IN ({placeholders})",
        node_ids,
    )
    ownership = {node_id: set[str]() for node_id in node_ids}
    for row in rows:
        values = _as_tuple(row)
        node_id = cast(int, values[0])
        file_path = cast(str, values[1])
        ownership[node_id].add(file_path)
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
        + "(created_at, solver_status, batch_count, oversized_count, max_spread, spread_json) "
        + "VALUES (?, ?, ?, ?, ?, ?)",
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


def get_latest_batch_plan(conn: sqlite3.Connection) -> BatchPlanRecord | None:
    row = fetchone(
        conn,
        "SELECT id, created_at, solver_status, batch_count, oversized_count, "
        + "max_spread, spread_json FROM batch_plans ORDER BY id DESC LIMIT 1",
    )
    if row is None:
        return None
    values = _as_tuple(row)
    return {
        "id": cast(int, values[0]),
        "created_at": cast(str, values[1]),
        "solver_status": cast(str, values[2]),
        "batch_count": cast(int, values[3]),
        "oversized_count": cast(int, values[4]),
        "max_spread": cast(int, values[5]),
        "spread_report": cast(list[dict[str, int | str]], json.loads(cast(str, values[6]))),
    }


def set_spec_hash(conn: sqlite3.Connection, spec_hash: str) -> None:
    now = datetime.now(UTC).isoformat()
    _ = conn.execute(
        "INSERT INTO plan_state (id, spec_hash, recorded_at) VALUES (1, ?, ?)"
        + " ON CONFLICT(id) DO UPDATE SET spec_hash = excluded.spec_hash,"
        + " recorded_at = excluded.recorded_at",
        (spec_hash, now),
    )
    conn.commit()


def get_spec_hash(conn: sqlite3.Connection) -> str | None:
    row = fetchone(conn, "SELECT spec_hash FROM plan_state WHERE id = 1")
    return cast(str, _as_tuple(row)[0]) if row else None


def drop_all(conn: sqlite3.Connection) -> int:
    count_row = fetchone(conn, "SELECT COUNT(*) FROM nodes")
    if count_row is None:
        raise RuntimeError("node count query returned no row")
    count = cast(int, _as_tuple(count_row)[0])
    for statement in (
        "DELETE FROM run_messages",
        "DELETE FROM runs",
        "DELETE FROM file_ownership",
        "DELETE FROM edges",
        "DELETE FROM goal_claims",
        "DELETE FROM node_reviews",
        "DELETE FROM nodes",
        "DELETE FROM plan_state",
        "DELETE FROM batch_plans",
    ):
        _ = conn.execute(statement)
    conn.commit()
    return count


def record_completion_duration(conn: sqlite3.Connection, node_id: int, duration: float) -> None:
    _ = conn.execute(
        "UPDATE nodes SET completion_duration_seconds = ? WHERE id = ?", (duration, node_id)
    )
    conn.commit()


def recent_completion_durations(conn: sqlite3.Connection, limit: int) -> list[float]:
    rows = fetchall(
        conn,
        "SELECT completion_duration_seconds FROM nodes "
        + "WHERE completion_duration_seconds IS NOT NULL "
        + "ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [cast(float, _as_tuple(row)[0]) for row in rows]


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
    _ = conn.execute(
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
    _ = conn.execute(
        "UPDATE nodes SET worktree_path = ?, branch_name = ? WHERE id = ? AND run_id = ?",
        (worktree_path, branch_name, node_id, run_id),
    )
    conn.commit()
