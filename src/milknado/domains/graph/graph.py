from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.common import (
    VALID_TRANSITIONS,
    MikadoEdge,
    MikadoNode,
    NodeStatus,
)


class MikadoGraph:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parent_id INTEGER,
                worktree_path TEXT,
                branch_name TEXT,
                run_id TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
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
        """)

    def _row_to_node(self, row: sqlite3.Row | tuple) -> MikadoNode:
        return MikadoNode(
            id=row[0],
            description=row[1],
            status=NodeStatus(row[2]),
            parent_id=row[3],
            worktree_path=row[4],
            branch_name=row[5],
            run_id=row[6],
            created_at=datetime.fromisoformat(row[7]),
            completed_at=datetime.fromisoformat(row[8]) if row[8] else None,
        )

    def add_node(
        self, description: str, parent_id: int | None = None
    ) -> MikadoNode:
        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO nodes (description, status, parent_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (description, NodeStatus.PENDING.value, parent_id, now),
        )
        self._conn.commit()
        node_id = cur.lastrowid
        assert node_id is not None

        if parent_id is not None:
            self.add_edge(parent_id, node_id)

        return self._row_to_node(
            self._conn.execute(
                "SELECT * FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
        )

    def add_edge(self, parent_id: int, child_id: int) -> MikadoEdge:
        if self._creates_cycle(parent_id, child_id):
            raise ValueError(
                f"Edge {parent_id}->{child_id} would create a cycle"
            )
        self._conn.execute(
            "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
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
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_all_nodes(self) -> list[MikadoNode]:
        rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_children(self, node_id: int) -> list[MikadoNode]:
        rows = self._conn.execute(
            "SELECT n.* FROM nodes n "
            "JOIN edges e ON n.id = e.child_id "
            "WHERE e.parent_id = ?",
            (node_id,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_leaves(self) -> list[MikadoNode]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE id NOT IN "
            "(SELECT DISTINCT parent_id FROM edges)"
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_ready_nodes(self) -> list[MikadoNode]:
        all_nodes = self.get_all_nodes()
        ready = []
        for node in all_nodes:
            children = self.get_children(node.id)
            if not children or all(
                c.status == NodeStatus.DONE for c in children
            ):
                ready.append(node)
        return ready

    def get_root(self) -> MikadoNode | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id NOT IN "
            "(SELECT DISTINCT child_id FROM edges)"
        ).fetchone()
        return self._row_to_node(row) if row else None

    def _transition_status(
        self, node_id: int, target: NodeStatus
    ) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        allowed = VALID_TRANSITIONS.get(node.status, set())
        if target not in allowed:
            raise ValueError(
                f"Cannot transition from {node.status.value} "
                f"to {target.value}"
            )
        completed_at = (
            datetime.now(UTC).isoformat()
            if target == NodeStatus.DONE
            else None
        )
        self._conn.execute(
            "UPDATE nodes SET status = ?, completed_at = ? WHERE id = ?",
            (target.value, completed_at, node_id),
        )
        self._conn.commit()

    def mark_done(self, node_id: int) -> None:
        self._transition_status(node_id, NodeStatus.DONE)

    def mark_failed(self, node_id: int) -> None:
        self._transition_status(node_id, NodeStatus.FAILED)
        self._conn.execute(
            "UPDATE nodes SET worktree_path = NULL, branch_name = NULL,"
            " run_id = NULL WHERE id = ?",
            (node_id,),
        )
        self._conn.commit()

    def mark_running(
        self,
        node_id: int,
        worktree_path: str | None = None,
        branch_name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._transition_status(node_id, NodeStatus.RUNNING)
        if worktree_path or branch_name or run_id:
            self._conn.execute(
                "UPDATE nodes SET worktree_path = ?, branch_name = ?, run_id = ? WHERE id = ?",
                (worktree_path, branch_name, run_id, node_id),
            )
            self._conn.commit()

    def set_run_id(self, node_id: int, run_id: str) -> None:
        self._conn.execute(
            "UPDATE nodes SET run_id = ? WHERE id = ?", (run_id, node_id),
        )
        self._conn.commit()

    def mark_pending(self, node_id: int) -> None:
        self._transition_status(node_id, NodeStatus.PENDING)
        self._conn.execute(
            "UPDATE nodes SET worktree_path = NULL, branch_name = NULL,"
            " run_id = NULL WHERE id = ?",
            (node_id,),
        )
        self._conn.commit()

    def mark_blocked(self, node_id: int) -> None:
        self._transition_status(node_id, NodeStatus.BLOCKED)

    def set_file_ownership(
        self, node_id: int, files: list[str]
    ) -> None:
        self._conn.execute(
            "DELETE FROM file_ownership WHERE node_id = ?", (node_id,)
        )
        self._conn.executemany(
            "INSERT INTO file_ownership (node_id, file_path) VALUES (?, ?)",
            [(node_id, f) for f in files],
        )
        self._conn.commit()

    def get_file_ownership(self, node_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT file_path FROM file_ownership WHERE node_id = ?",
            (node_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def check_parallel_safety(
        self, node_ids: list[int]
    ) -> list[tuple[int, int, list[str]]]:
        ownership: dict[int, set[str]] = {}
        for nid in node_ids:
            ownership[nid] = set(self.get_file_ownership(nid))

        conflicts: list[tuple[int, int, list[str]]] = []
        ids = list(node_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                overlap = ownership[ids[i]] & ownership[ids[j]]
                if overlap:
                    conflicts.append(
                        (ids[i], ids[j], sorted(overlap))
                    )
        return conflicts

    def close(self) -> None:
        self._conn.close()
