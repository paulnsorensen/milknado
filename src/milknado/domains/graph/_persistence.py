"""DB schema creation, row serialization, and admin helpers for MikadoGraph."""

from __future__ import annotations

import itertools
import json
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus

if TYPE_CHECKING:
    from milknado.domains.batching import BatchPlan


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
            kind TEXT NOT NULL DEFAULT 'task'
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
