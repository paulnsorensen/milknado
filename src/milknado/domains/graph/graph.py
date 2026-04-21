from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common import (
    VALID_TRANSITIONS,
    MikadoEdge,
    MikadoNode,
    NodeStatus,
)
from milknado.domains.graph._persistence import (
    check_parallel_safety,
    create_tables,
    drop_all,
    ensure_schema,
    get_file_ownership,
    get_latest_batch_plan,
    get_spec_hash,
    recent_completion_durations,
    record_batch_plan,
    record_completion_duration,
    row_to_node,
    set_dispatched_at,
    set_file_ownership,
    set_spec_hash,
)

if TYPE_CHECKING:
    from milknado.domains.batching import BatchPlan


class MikadoGraph:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        create_tables(self._conn)
        ensure_schema(self._conn)

    def add_node(
        self,
        description: str,
        parent_id: int | None = None,
        *,
        oversized: bool = False,
        batch_index: int | None = None,
    ) -> MikadoNode:
        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO nodes "
            "(description, status, parent_id, created_at, oversized, batch_index) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (description, NodeStatus.PENDING.value, parent_id, now,
             1 if oversized else 0, batch_index),
        )
        self._conn.commit()
        node_id = cur.lastrowid
        assert node_id is not None
        if parent_id is not None:
            self.add_edge(parent_id, node_id)
        return row_to_node(
            self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        )

    def set_batch_metadata(self, node_id: int, oversized: bool, batch_index: int | None) -> None:
        cur = self._conn.execute(
            "UPDATE nodes SET oversized = ?, batch_index = ? WHERE id = ?",
            (1 if oversized else 0, batch_index, node_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Node {node_id} not found")
        self._conn.commit()

    def set_parent_id(self, node_id: int, parent_id: int | None) -> None:
        """Update parent_id without creating an edge (used by batching bridge)."""
        cur = self._conn.execute(
            "UPDATE nodes SET parent_id = ? WHERE id = ?", (parent_id, node_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Node {node_id} not found")
        self._conn.commit()

    def add_edge(self, parent_id: int, child_id: int) -> MikadoEdge:
        if self._creates_cycle(parent_id, child_id):
            raise ValueError(f"Edge {parent_id}->{child_id} would create a cycle")
        self._conn.execute(
            "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)", (parent_id, child_id),
        )
        self._conn.commit()
        return MikadoEdge(parent_id=parent_id, child_id=child_id)

    def _creates_cycle(self, parent_id: int, child_id: int) -> bool:
        visited: set[int] = set()
        stack = [parent_id]
        while stack:
            current = stack.pop()
            if current == child_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            rows = self._conn.execute(
                "SELECT parent_id FROM edges WHERE child_id = ?", (current,)
            ).fetchall()
            stack.extend(row[0] for row in rows)
        return False

    def get_node(self, node_id: int) -> MikadoNode | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return row_to_node(row) if row else None

    def get_all_nodes(self) -> list[MikadoNode]:
        return [row_to_node(r) for r in self._conn.execute("SELECT * FROM nodes").fetchall()]

    def get_children(self, node_id: int) -> list[MikadoNode]:
        rows = self._conn.execute(
            "SELECT n.* FROM nodes n JOIN edges e ON n.id = e.child_id WHERE e.parent_id = ?",
            (node_id,),
        ).fetchall()
        return [row_to_node(r) for r in rows]

    def get_leaves(self) -> list[MikadoNode]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT parent_id FROM edges)"
        ).fetchall()
        return [row_to_node(r) for r in rows]

    def get_ready_nodes(self) -> list[MikadoNode]:
        root = self.get_root()
        ready = []
        for node in self.get_all_nodes():
            if node.status != NodeStatus.PENDING:
                continue
            if root is not None and node.id == root.id:
                continue
            children = self.get_children(node.id)
            if not children or all(c.status == NodeStatus.DONE for c in children):
                ready.append(node)
        return ready

    def get_root(self) -> MikadoNode | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT child_id FROM edges)"
        ).fetchone()
        return row_to_node(row) if row else None

    def _assert_transition(self, node_id: int, target: NodeStatus) -> None:
        from milknado.domains.common.errors import InvalidTransition
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        allowed = VALID_TRANSITIONS.get(node.status, set())
        if target not in allowed:
            raise InvalidTransition(
                node_id=node_id,
                current=node.status,
                target=target,
                valid_targets=tuple(allowed),
            )

    def _transition_status(self, node_id: int, target: NodeStatus) -> None:
        self._assert_transition(node_id, target)
        completed_at = datetime.now(UTC).isoformat() if target == NodeStatus.DONE else None
        self._conn.execute(
            "UPDATE nodes SET status = ?, completed_at = ? WHERE id = ?",
            (target.value, completed_at, node_id),
        )
        self._conn.commit()

    def mark_done(self, node_id: int) -> None:
        self._transition_status(node_id, NodeStatus.DONE)

    def mark_failed(self, node_id: int) -> None:
        self._assert_transition(node_id, NodeStatus.FAILED)
        self._conn.execute(
            "UPDATE nodes SET status = ?, completed_at = NULL, "
            "worktree_path = NULL, branch_name = NULL, run_id = NULL WHERE id = ?",
            (NodeStatus.FAILED.value, node_id),
        )
        self._conn.commit()

    def mark_running(
        self,
        node_id: int,
        worktree_path: str | None = None,
        branch_name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._assert_transition(node_id, NodeStatus.RUNNING)
        self._conn.execute(
            "UPDATE nodes SET status = ?, completed_at = NULL, "
            "worktree_path = ?, branch_name = ?, run_id = ? WHERE id = ?",
            (NodeStatus.RUNNING.value, worktree_path, branch_name, run_id, node_id),
        )
        self._conn.commit()

    def set_run_id(self, node_id: int, run_id: str) -> None:
        cur = self._conn.execute(
            "UPDATE nodes SET run_id = ? WHERE id = ?",
            (run_id, node_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Node {node_id} not found")
        self._conn.commit()

    def mark_pending(self, node_id: int) -> None:
        self._assert_transition(node_id, NodeStatus.PENDING)
        self._conn.execute(
            "UPDATE nodes SET status = ?, completed_at = NULL, "
            "worktree_path = NULL, branch_name = NULL, run_id = NULL WHERE id = ?",
            (NodeStatus.PENDING.value, node_id),
        )
        self._conn.commit()

    def mark_blocked(self, node_id: int) -> None:
        self._transition_status(node_id, NodeStatus.BLOCKED)

    def set_file_ownership(self, node_id: int, files: list[str]) -> None:
        set_file_ownership(self._conn, node_id, files)

    def get_file_ownership(self, node_id: int) -> list[str]:
        return get_file_ownership(self._conn, node_id)

    def check_parallel_safety(self, node_ids: list[int]) -> list[tuple[int, int, list[str]]]:
        return check_parallel_safety(self._conn, node_ids)

    def record_batch_plan(self, plan: BatchPlan) -> int:
        return record_batch_plan(self._conn, plan)

    def get_latest_batch_plan(self) -> dict | None:
        return get_latest_batch_plan(self._conn)

    def _record_completion_duration(self, node_id: int, duration_seconds: float) -> None:
        record_completion_duration(self._conn, node_id, duration_seconds)

    def recent_completion_durations(self, limit: int) -> list[float]:
        return recent_completion_durations(self._conn, limit)

    def set_dispatched_at(self, node_id: int) -> None:
        set_dispatched_at(self._conn, node_id)

    def set_spec_hash(self, spec_hash: str) -> None:
        set_spec_hash(self._conn, spec_hash)

    def get_spec_hash(self) -> str | None:
        return get_spec_hash(self._conn)

    def drop_all(self) -> int:
        return drop_all(self._conn)

    def close(self) -> None:
        self._conn.close()
