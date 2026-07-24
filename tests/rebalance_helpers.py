"""Tests for the rebalance engine — acceptance #11-#15.

Domain functions run against in-memory sqlite + create_tables; the app layer
runs against a tmp_path project root with a fake GitPort at the boundary.
"""

from __future__ import annotations

import sqlite3
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest

from milknado.adapters.git import GitAdapter
from milknado.app.rebalance import RebalanceOptions
from milknado.app.rebalance import rebalance as _run_rebalance
from milknado.domains.common.errors import GitOperationError
from milknado.domains.graph import _rebalance as graph_rebalance
from milknado.domains.graph._persistence import create_tables, migrate
from milknado.domains.graph.rebalance import (
    RebalanceReport,
)

NOW = "2026-07-24T00:00:00+00:00"

find_archivable_roots = graph_rebalance._find_archivable_roots
structural_report = graph_rebalance.structural_report
sweep_archivable = graph_rebalance._sweep_archivable


def group_orphans(conn: sqlite3.Connection) -> int:
    return graph_rebalance._group_orphans(conn)[0]


def run_rebalance(
    project_root: Path,
    git: FakeGit | GitAdapter | None = None,
    **passes: bool,
) -> RebalanceReport:
    options = RebalanceOptions(**passes)
    return _run_rebalance(project_root, options, git)


@pytest.fixture()
def conn() -> Generator[sqlite3.Connection, None, None]:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    create_tables(c)
    yield c
    c.close()


def _insert_node(
    conn: sqlite3.Connection,
    description: str,
    status: str,
    *,
    kind: str = "task",
    parent_id: int | None = None,
    archived_at: str | None = None,
    worktree_path: str | None = None,
    branch_name: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO nodes (description, status, parent_id, kind, created_at, archived_at, "
        "worktree_path, branch_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (description, status, parent_id, kind, NOW, archived_at, worktree_path, branch_name),
    )
    node_id = cur.lastrowid
    assert node_id is not None
    if parent_id is not None:
        conn.execute("INSERT INTO edges (parent_id, child_id) VALUES (?, ?)", (parent_id, node_id))
    return node_id


def _register_run(conn: sqlite3.Connection, node_id: int, status: str = "completed") -> None:
    conn.execute(
        "INSERT INTO runs (run_id, node_id, status, log_path, started_at) VALUES (?, ?, ?, '', ?)",
        (f"run-{node_id}", node_id, status, NOW),
    )


def _archived_ids(conn: sqlite3.Connection) -> set[int]:
    return {r[0] for r in conn.execute("SELECT id FROM nodes WHERE archived_at IS NOT NULL")}


class FakeGit:
    """Records teardown calls; configured failures raise like GitAdapter does."""

    def __init__(self) -> None:
        self.removed: list[Path] = []
        self.deleted_branches: list[str] = []
        self.pruned = 0
        self.fail_remove: set[str] = set()
        self.fail_delete: set[str] = set()

    def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
        if str(path) in self.fail_remove:
            raise GitOperationError("worktree remove", "simulated teardown failure")
        self.removed.append(path)

    def delete_branch(self, branch: str) -> None:
        if branch in self.fail_delete:
            raise GitOperationError("branch -d", "branch not fully merged")
        self.deleted_branches.append(branch)

    def prune_worktrees(self) -> None:
        self.pruned += 1

    def worktree_teardown_blocker(self, path: Path, target: str = "HEAD") -> str | None:
        if str(path) in self.fail_remove:
            return "simulated teardown failure"
        return None


def _project_with_db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / ".milknado" / "milknado.db"
    db_path.parent.mkdir(exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    create_tables(connection)
    migrate(connection)
    return connection


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )
