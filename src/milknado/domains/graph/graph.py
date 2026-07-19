from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common import (
    MikadoEdge,
    MikadoNode,
    NodeKind,
    NodeSpec,
    NodeStatus,
    RunResult,
)
from milknado.domains.common.types import BUILTIN_FLAVORS
from milknado.domains.graph import _creation, _mutations, _persistence, _reads, _status
from milknado.domains.graph._analytics_facade import _AnalyticsFacade
from milknado.domains.graph._pipeline import StatusPipeline, _PluginAsMiddleware

if TYPE_CHECKING:
    from milknado.domains.common import PluginHook

_logger = logging.getLogger(__name__)


class MikadoGraph(_AnalyticsFacade):
    """Thin facade over the graph slice's free-function modules.

    Public methods delegate to `_reads`/`_creation`/`_status`/`_persistence`/
    `_mutations`; status transitions run through a `StatusPipeline` so plugins
    observe before/after each change. Rich docstrings live on the free functions.
    """

    def __init__(self, db_path: Path, plugins: Sequence[PluginHook] = ()) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # Explicit so the concurrent-writer wait window (detached runner + server
        # both writing the same db) is documented, not implicit in connect()'s default.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._pipeline = StatusPipeline([_PluginAsMiddleware(p) for p in plugins])
        _persistence.create_tables(self._conn)

    # ── Creation ─────────────────────────────────────────────────────────────

    def add_node(
        self, description: str, parent_id: int | None = None, spec: NodeSpec | None = None
    ) -> MikadoNode:
        return _creation.add_node(self._conn, description, parent_id, spec or NodeSpec())

    def add_edge(self, parent_id: int, child_id: int) -> MikadoEdge:
        return _creation.add_edge(self._conn, parent_id, child_id)

    def set_batch_metadata(self, node_id: int, oversized: bool, batch_index: int | None) -> None:
        _creation.set_batch_metadata(self._conn, node_id, oversized, batch_index)

    def set_parent_id(self, node_id: int, parent_id: int | None) -> None:
        _creation.set_parent_id(self._conn, node_id, parent_id)

    def _creates_cycle(self, parent_id: int, child_id: int) -> bool:
        return _mutations.would_create_cycle(self._conn, parent_id, child_id)

    # ── Mutations ────────────────────────────────────────────────────────────

    def delete_node(self, node_id: int, cascade: bool = False) -> int:
        return _mutations.delete_subtree(self._conn, node_id, cascade)

    def update_node(
        self,
        node_id: int,
        description: str | None = None,
        kind: NodeKind | None = None,
        flavor: str | None = None,
        artifact_path: str | None = None,
        flavor_registry: frozenset[str] = BUILTIN_FLAVORS,
    ) -> None:
        _mutations.update_node_fields(
            self._conn, node_id, description, kind, flavor, artifact_path, flavor_registry
        )

    def move_node(self, node_id: int, new_parent_id: int | None) -> None:
        _mutations.reparent(self._conn, node_id, new_parent_id)

    # ── Reads ────────────────────────────────────────────────────────────────

    def get_node(self, node_id: int) -> MikadoNode | None:
        return _reads.get_node(self._conn, node_id)

    def get_nodes(self, node_ids: Iterable[int]) -> list[MikadoNode]:
        return _reads.get_nodes(self._conn, node_ids)

    def latest_results_for_nodes(self, node_ids: Iterable[int]) -> dict[int, str]:
        return _reads.latest_results_for_nodes(self._conn, node_ids)

    def find_node_by_wiki_ref(self, wiki_ref: str) -> MikadoNode | None:
        return _reads.find_node_by_wiki_ref(self._conn, wiki_ref)

    def find_node_by_github_ref(self, github_ref: str) -> MikadoNode | None:
        return _reads.find_node_by_github_ref(self._conn, github_ref)

    def get_all_nodes(self) -> list[MikadoNode]:
        return _reads.get_all_nodes(self._conn)

    def get_children(self, node_id: int) -> list[MikadoNode]:
        return _reads.get_children(self._conn, node_id)

    def get_children_map(self) -> dict[int, list[MikadoNode]]:
        return _reads.get_children_map(self._conn)

    def get_leaves(self) -> list[MikadoNode]:
        return _reads.get_leaves(self._conn)

    def get_ready_nodes(
        self, *, kind: NodeKind | None = None, flavor: str | None = None, limit: int = 100
    ) -> list[MikadoNode]:
        return _reads.get_ready_nodes(self._conn, kind=kind, flavor=flavor, limit=limit)

    def get_node_summaries(
        self,
        *,
        status: NodeStatus | None = None,
        kind: NodeKind | None = None,
        flavor: str | None = None,
        page: tuple[int, int] = (100, 0),
    ) -> list[dict[str, int | str]]:
        return _reads.get_node_summaries(
            self._conn, status=status, kind=kind, flavor=flavor, page=page
        )

    def get_root(self) -> MikadoNode | None:
        return _reads.get_root(self._conn)

    def get_roots(self) -> list[MikadoNode]:
        return _reads.get_roots(self._conn)

    def get_next_runnable(self, kind: NodeKind | None = None) -> MikadoNode | None:
        ready = _reads.get_ready_nodes(self._conn, kind=kind, limit=1)
        return ready[0] if ready else None

    def _node_status(self, node_id: int) -> NodeStatus | None:
        return _reads.node_status(self._conn, node_id)

    # ── Status transitions (pipeline-driven) ─────────────────────────────────

    def _transition_status(self, node_id: int, target: NodeStatus) -> None:
        _status.transition_status(self._pipeline, self._conn, node_id, target)

    def mark_done(self, node_id: int) -> None:
        _status.mark_done(self._pipeline, self._conn, node_id)

    def mark_failed(self, node_id: int) -> None:
        _status.mark_failed(self._pipeline, self._conn, node_id)

    def mark_running(
        self,
        node_id: int,
        worktree_path: str | None = None,
        branch_name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        _status.mark_running(
            self._pipeline, self._conn, node_id, worktree_path, branch_name, run_id
        )

    def mark_pending(self, node_id: int) -> None:
        _status.mark_pending(self._pipeline, self._conn, node_id)

    def mark_blocked(self, node_id: int) -> None:
        _status.mark_blocked(self._pipeline, self._conn, node_id)

    def complete_root(self) -> bool:
        return _status.complete_root(self._pipeline, self._conn)

    def claim_node(self, node_id: int, run_id: str, *, now: str) -> bool:
        return _status.claim_node(self._pipeline, self._conn, node_id, run_id, now=now)

    def release(self, node_id: int, run_id: str) -> bool:
        return _status.release(self._pipeline, self._conn, node_id, run_id)

    def mark_terminal(self, node_id: int, run_id: str, status: NodeStatus) -> bool:
        return _status.mark_terminal(self._pipeline, self._conn, node_id, run_id, status)

    def try_reclaim(self, node_id: int, *, now: str) -> bool:
        return _status.try_reclaim(self._pipeline, self._conn, node_id, now=now)

    # ── Dispatch orchestration ───────────────────────────────────────────────

    def claim_ancestor_goal_for_dispatch(
        self, node_id: int, run_id: str, *, now: str | None = None
    ) -> None:
        """Enforce and claim the ancestor-goal subtree fence before dispatch.

        Raises ValueError if an ancestor goal is claimed by a different live
        coordinator (a different pid). Same-process dispatch (same pid) is exempt;
        no-op when there is no ancestor goal.
        """
        claim = self.ancestor_goal_claimed_by_other(node_id)
        if claim is not None and claim["pid"] != os.getpid():
            raise ValueError(
                f"node {node_id}'s ancestor goal is claimed by a different run "
                f"({claim['run_id']!r}); dispatch refused"
            )
        self.claim_ancestor_goal(
            node_id, run_id, os.getpid(), now=now or datetime.now(UTC).isoformat()
        )

    def claim_node_for_dispatch(self, node_id: int, run_id: str, *, now: str) -> None:
        """Fence the ancestor-goal subtree, then atomically claim node_id as RUNNING.

        Raises ValueError if blocked by a foreign ancestor-goal claim, or if
        node_id itself could not be claimed.
        """
        self.claim_ancestor_goal_for_dispatch(node_id, run_id, now=now)
        if not self.claim_node(node_id, run_id, now=now):
            current = self.get_node(node_id)
            status = current.status.value if current is not None else "gone"
            raise ValueError(
                f"node {node_id} is already {status}; set status back to pending to retry"
            )

    # ── Node fields (refs, pid, worktree) ────────────────────────────────────

    def set_wiki_ref(self, node_id: int, wiki_ref: str) -> None:
        """Set the deterministic wiki key so an orphan goal node round-trips on export."""
        self._update_node_field("wiki_ref", wiki_ref, node_id)

    def set_github_ref(self, node_id: int, github_ref: str) -> None:
        """Bind the GitHub Projects node id recorded when this goal is linked/bound."""
        self._update_node_field("github_ref", github_ref, node_id)

    def set_run_id(self, node_id: int, run_id: str) -> None:
        self._update_node_field("run_id", run_id, node_id)

    def _update_node_field(self, column: str, value: str, node_id: int) -> None:
        # column is an internal literal (never user input) — safe to interpolate.
        cur = self._conn.execute(
            f"UPDATE nodes SET {column} = ? WHERE id = ?",  # noqa: S608
            (value, node_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Node {node_id} not found")
        self._conn.commit()

    def set_pid(self, node_id: int, run_id: str, pid: int) -> None:
        _persistence.set_pid(self._conn, node_id, run_id, pid)

    def set_worktree(
        self, node_id: int, run_id: str, worktree_path: str, branch_name: str
    ) -> None:
        _persistence.set_worktree(self._conn, node_id, run_id, worktree_path, branch_name)

    # ── Run repository ───────────────────────────────────────────────────────

    def start_run(
        self,
        run_id: str,
        node_id: int,
        log_path: str,
        started_at: str,
        timeout_seconds: int | None,
        pid: int | None = None,
    ) -> None:
        _persistence.start_run(
            self._conn, run_id, node_id, log_path, started_at, timeout_seconds, pid
        )

    def finish_run(self, run_id: str, result: RunResult) -> None:
        _persistence.finish_run(self._conn, run_id, result)

    def set_run_pid(self, run_id: str, pid: int) -> None:
        _persistence.set_run_pid(self._conn, run_id, pid)

    def get_run(self, run_id: str) -> dict | None:
        return _persistence.get_run(self._conn, run_id)

    def runs_for_node(
        self, node_id: int, *, terminal_only: bool = False, run_id: str | None = None
    ) -> list[dict]:
        return _persistence.runs_for_node(
            self._conn, node_id, terminal_only=terminal_only, run_id=run_id
        )

    def recent_runs(self, limit: int) -> list[dict]:
        return _persistence.recent_runs(self._conn, limit)

    def deposit_run_message(self, run_id: str, role: str, body: str, created_at: str) -> int:
        return _persistence.deposit_run_message(self._conn, run_id, role, body, created_at)

    def latest_run_message(self, run_id: str, role: str) -> str | None:
        return _persistence.latest_run_message(self._conn, run_id, role)

    # ── File ownership ───────────────────────────────────────────────────────

    def set_file_ownership(self, node_id: int, files: list[str]) -> None:
        _persistence.set_file_ownership(self._conn, node_id, files)

    def get_file_ownership(self, node_id: int) -> list[str]:
        return _persistence.get_file_ownership(self._conn, node_id)

    def get_file_ownership_map(self) -> dict[int, list[str]]:
        return _persistence.get_file_ownership_map(self._conn)

    def check_parallel_safety(self, node_ids: list[int]) -> list[tuple[int, int, list[str]]]:
        return _persistence.check_parallel_safety(self._conn, node_ids)

    def drop_all(self) -> int:
        return _persistence.drop_all(self._conn)

    # ── Goal-claim fencing ───────────────────────────────────────────────────

    def claim_or_reclaim_goal(self, goal_id: int, owner: str, pid: int, *, now: str) -> bool:
        return _persistence.claim_or_reclaim_goal(self._conn, goal_id, owner, pid, now=now)

    def claim_goal(self, goal_id: int, run_id: str, *, now: str) -> bool:
        node = self.get_node(goal_id)
        if node is None:
            raise ValueError(f"node {goal_id} not found")
        if node.kind != NodeKind.GOAL:
            raise ValueError(
                f"node {goal_id} has kind={node.kind.value}; only goal nodes can be claimed"
            )
        return _persistence.claim_goal_row(self._conn, goal_id, run_id, now)

    def release_goal(self, goal_id: int, run_id: str) -> bool:
        return _persistence.release_goal_row(self._conn, goal_id, run_id)

    def claim_ancestor_goal(self, node_id: int, run_id: str, pid: int, *, now: str) -> int | None:
        goal_id = _persistence.find_ancestor_goal_id(self._conn, node_id)
        if goal_id is None:
            return None
        _persistence.claim_goal_row(self._conn, goal_id, run_id, now, pid=pid)
        return goal_id

    def set_goal_pid(self, goal_id: int, run_id: str, pid: int) -> None:
        _persistence.set_goal_claim_pid(self._conn, goal_id, run_id, pid)

    def try_reclaim_goal(self, goal_id: int, *, now: str) -> bool:
        return _persistence.try_reclaim_goal(self._conn, goal_id, now=now)

    def ancestor_goal_claimed_by_other(
        self, node_id: int, *, caller_run_id: str | None = None
    ) -> dict | None:
        return _persistence.ancestor_goal_claimed_by_other(
            self._conn, node_id, caller_run_id=caller_run_id
        )

    def close(self) -> None:
        # Fold the WAL into the main .db file so the documented
        # `git add -f .milknado/milknado.db` persistence step captures every
        # committed write. A non-last-connection close does no implicit
        # checkpoint, so without this an MCP tool call or detached run can leave
        # its tail in the -wal sidecar and lose it on container reclaim.
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            _logger.warning("WAL checkpoint on close failed", exc_info=True)
        finally:
            self._conn.close()
