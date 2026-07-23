from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
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
from milknado.domains.graph import (
    _creation,
    _goal_claims,
    _mutations,
    _persistence,
    _reads,
    _status,
)
from milknado.domains.graph._analytics_facade import _AnalyticsFacade, _synchronized
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
        self._lock = RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # Explicit so the concurrent-writer wait window (detached runner + server
        # both writing the same db) is documented, not implicit in connect()'s default.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._pipeline = StatusPipeline([_PluginAsMiddleware(p) for p in plugins])
        _persistence.create_tables(self._conn)

    # ── Creation ─────────────────────────────────────────────────────────────

    @_synchronized
    def add_node(
        self, description: str, parent_id: int | None = None, spec: NodeSpec | None = None
    ) -> MikadoNode:
        return _creation.add_node(self._conn, description, parent_id, spec or NodeSpec())

    @_synchronized
    def add_edge(self, parent_id: int, child_id: int) -> MikadoEdge:
        return _creation.add_edge(self._conn, parent_id, child_id)

    @_synchronized
    def set_batch_metadata(self, node_id: int, oversized: bool, batch_index: int | None) -> None:
        _creation.set_batch_metadata(self._conn, node_id, oversized, batch_index)

    @_synchronized
    def set_parent_id(self, node_id: int, parent_id: int | None) -> None:
        _creation.set_parent_id(self._conn, node_id, parent_id)

    def _creates_cycle(self, parent_id: int, child_id: int) -> bool:
        return _mutations.would_create_cycle(self._conn, parent_id, child_id)

    # ── Mutations ────────────────────────────────────────────────────────────

    @_synchronized
    def delete_node(self, node_id: int, cascade: bool = False) -> int:
        return _mutations.delete_subtree(self._conn, node_id, cascade)

    @_synchronized
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

    @_synchronized
    def move_node(self, node_id: int, new_parent_id: int | None) -> None:
        _mutations.reparent(self._conn, node_id, new_parent_id)

    # ── Reads ────────────────────────────────────────────────────────────────

    @_synchronized
    def get_node(self, node_id: int) -> MikadoNode | None:
        return _reads.get_node(self._conn, node_id)

    @_synchronized
    def get_nodes(self, node_ids: Iterable[int]) -> list[MikadoNode]:
        return _reads.get_nodes(self._conn, node_ids)

    @_synchronized
    def latest_results_for_nodes(self, node_ids: Iterable[int]) -> dict[int, str]:
        return _reads.latest_results_for_nodes(self._conn, node_ids)

    @_synchronized
    def find_node_by_wiki_ref(self, wiki_ref: str) -> MikadoNode | None:
        return _reads.find_node_by_wiki_ref(self._conn, wiki_ref)

    @_synchronized
    def find_node_by_github_ref(self, github_ref: str) -> MikadoNode | None:
        return _reads.find_node_by_github_ref(self._conn, github_ref)

    @_synchronized
    def get_all_nodes(self) -> list[MikadoNode]:
        return _reads.get_all_nodes(self._conn)

    @_synchronized
    def get_children(self, node_id: int) -> list[MikadoNode]:
        return _reads.get_children(self._conn, node_id)

    @_synchronized
    def get_children_map(self) -> dict[int, list[MikadoNode]]:
        return _reads.get_children_map(self._conn)

    @_synchronized
    def get_leaves(self) -> list[MikadoNode]:
        return _reads.get_leaves(self._conn)

    @_synchronized
    def get_ready_nodes(
        self, *, kind: NodeKind | None = None, flavor: str | None = None, limit: int = 100
    ) -> list[MikadoNode]:
        return _reads.get_ready_nodes(self._conn, kind=kind, flavor=flavor, limit=limit)

    @_synchronized
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

    @_synchronized
    def get_root(self) -> MikadoNode | None:
        return _reads.get_root(self._conn)

    @_synchronized
    def get_roots(self) -> list[MikadoNode]:
        return _reads.get_roots(self._conn)

    @_synchronized
    def get_next_runnable(self, kind: NodeKind | None = None) -> MikadoNode | None:
        ready = _reads.get_ready_nodes(self._conn, kind=kind, limit=1)
        return ready[0] if ready else None

    @_synchronized
    def get_execution_overview(
        self,
        node_ids: Iterable[int],
        excluded_node_ids: Iterable[int] = (),
    ) -> tuple[str, dict[int, str], int]:
        from milknado.domains.execution.executor import get_dispatchable_nodes

        self._conn.execute("SAVEPOINT execution_overview")
        try:
            root = _reads.get_root(self._conn)
            descriptions = {
                node.id: node.description for node in _reads.get_nodes(self._conn, node_ids)
            }
            excluded = set(excluded_node_ids)
            available = sum(node_id not in excluded for node_id in get_dispatchable_nodes(self))
            return root.description if root is not None else "", descriptions, available
        finally:
            self._conn.execute("RELEASE SAVEPOINT execution_overview")

    def _node_status(self, node_id: int) -> NodeStatus | None:
        return _reads.node_status(self._conn, node_id)

    # ── Status transitions (pipeline-driven) ─────────────────────────────────

    def _transition_status(self, node_id: int, target: NodeStatus) -> None:
        _status.transition_status(self._pipeline, self._conn, node_id, target)

    @_synchronized
    def mark_done(self, node_id: int) -> None:
        _status.mark_done(self._pipeline, self._conn, node_id)

    @_synchronized
    def mark_failed(self, node_id: int) -> None:
        _status.mark_failed(self._pipeline, self._conn, node_id)

    @_synchronized
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

    @_synchronized
    def mark_pending(self, node_id: int) -> None:
        _status.mark_pending(self._pipeline, self._conn, node_id)

    @_synchronized
    def mark_blocked(self, node_id: int) -> None:
        _status.mark_blocked(self._pipeline, self._conn, node_id)

    @_synchronized
    def complete_root(self) -> bool:
        return _status.complete_root(self._pipeline, self._conn)

    @_synchronized
    def claim_node(self, node_id: int, run_id: str, *, now: str, pid: int | None = None) -> bool:
        return _status.claim_node(self._pipeline, self._conn, node_id, run_id, now=now, pid=pid)

    @_synchronized
    def release(self, node_id: int, run_id: str) -> bool:
        return _status.release(self._pipeline, self._conn, node_id, run_id)

    @_synchronized
    def mark_terminal(self, node_id: int, run_id: str, status: NodeStatus) -> bool:
        return _status.mark_terminal(self._pipeline, self._conn, node_id, run_id, status)

    @_synchronized
    def mark_blocked_fenced(self, node_id: int, run_id: str) -> bool:
        """Block the owning run without releasing its pinned worktree."""
        return _status.mark_blocked_fenced(self._pipeline, self._conn, node_id, run_id)

    @_synchronized
    def try_reclaim(self, node_id: int, *, now: str) -> bool:
        return _status.try_reclaim(self._pipeline, self._conn, node_id, now=now)

    # ── Dispatch orchestration ───────────────────────────────────────────────

    @_synchronized
    def claim_ancestor_goal_for_dispatch(
        self,
        node_id: int,
        run_id: str,
        *,
        now: str | None = None,
        pid: int | None = None,
    ) -> None:
        """Enforce and claim the ancestor-goal subtree fence before dispatch."""
        owner_pid = os.getpid() if pid is None else pid
        claim = self.ancestor_goal_claimed_by_other(node_id)
        if claim is not None and claim["pid"] != owner_pid:
            raise ValueError(
                f"node {node_id}'s ancestor goal is claimed by a different run "
                f"({claim['run_id']!r}); dispatch refused"
            )
        self.claim_ancestor_goal(
            node_id, run_id, owner_pid, now=now or datetime.now(UTC).isoformat()
        )

    @_synchronized
    def claim_node_for_dispatch(
        self, node_id: int, run_id: str, *, now: str, pid: int | None = None
    ) -> None:
        """Fence the ancestor-goal subtree, then atomically claim node_id as RUNNING."""
        owner_pid = os.getpid() if pid is None else pid
        self.claim_ancestor_goal_for_dispatch(node_id, run_id, now=now, pid=owner_pid)
        if not self.claim_node(node_id, run_id, now=now, pid=owner_pid):
            current = self.get_node(node_id)
            status = current.status.value if current is not None else "gone"
            raise ValueError(
                f"node {node_id} is already {status}; set status back to pending to retry"
            )

    # ── Node fields (refs, pid, worktree) ────────────────────────────────────

    def _update_node_field(self, column: str, value: str, node_id: int) -> None:
        cur = self._conn.execute(
            f"UPDATE nodes SET {column} = ? WHERE id = ?",  # noqa: S608
            (value, node_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Node {node_id} not found")
        self._conn.commit()

    @_synchronized
    def set_wiki_ref(self, node_id: int, wiki_ref: str) -> None:
        """Set the deterministic wiki key so an orphan goal node round-trips on export."""
        self._update_node_field("wiki_ref", wiki_ref, node_id)

    @_synchronized
    def set_github_ref(self, node_id: int, github_ref: str) -> None:
        """Bind the GitHub Projects node id recorded when this goal is linked/bound."""
        self._update_node_field("github_ref", github_ref, node_id)

    @_synchronized
    def set_run_id(self, node_id: int, run_id: str) -> None:
        self._update_node_field("run_id", run_id, node_id)

    @_synchronized
    def replace_run_id(self, node_id: int, expected_run_id: str, run_id: str) -> bool:
        """Replace a provisional dispatch fence without crossing an owner."""
        cur = self._conn.execute(
            "UPDATE nodes SET run_id = ? WHERE id = ? AND status = 'running' AND run_id = ?",
            (run_id, node_id, expected_run_id),
        )
        self._conn.commit()
        return cur.rowcount == 1

    @_synchronized
    def set_pid(self, node_id: int, run_id: str, pid: int) -> None:
        _persistence.set_pid(self._conn, node_id, run_id, pid)

    @_synchronized
    def set_worktree(
        self, node_id: int, run_id: str, worktree_path: str, branch_name: str
    ) -> None:
        _persistence.set_worktree(self._conn, node_id, run_id, worktree_path, branch_name)

    @_synchronized
    def check_parallel_safety(self, node_ids: list[int]) -> list[tuple[int, int, list[str]]]:
        return _persistence.check_parallel_safety(self._conn, node_ids)

    @_synchronized
    def drop_all(self) -> int:
        return _persistence.drop_all(self._conn)

    # ── Run repository ───────────────────────────────────────────────────────

    @_synchronized
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

    @_synchronized
    def finish_run(self, run_id: str, result: RunResult) -> bool:
        return _persistence.finish_run(self._conn, run_id, result)

    @_synchronized
    def set_run_pid(self, run_id: str, pid: int) -> None:
        _persistence.set_run_pid(self._conn, run_id, pid)

    @_synchronized
    def get_run(self, run_id: str) -> dict | None:
        return _persistence.get_run(self._conn, run_id)

    @_synchronized
    def runs_for_node(
        self, node_id: int, *, terminal_only: bool = False, run_id: str | None = None
    ) -> list[dict]:
        return _persistence.runs_for_node(
            self._conn, node_id, terminal_only=terminal_only, run_id=run_id
        )

    @_synchronized
    def recent_runs(self, limit: int) -> list[dict]:
        return _persistence.recent_runs(self._conn, limit)

    @_synchronized
    def deposit_run_message(self, run_id: str, role: str, body: str, created_at: str) -> int:
        return _persistence.deposit_run_message(self._conn, run_id, role, body, created_at)

    @_synchronized
    def latest_run_message(self, run_id: str, role: str) -> str | None:
        return _persistence.latest_run_message(self._conn, run_id, role)

    # ── File ownership ───────────────────────────────────────────────────────

    @_synchronized
    def set_file_ownership(self, node_id: int, files: list[str]) -> None:
        _persistence.set_file_ownership(self._conn, node_id, files)

    @_synchronized
    def get_file_ownership(self, node_id: int) -> list[str]:
        return _persistence.get_file_ownership(self._conn, node_id)

    @_synchronized
    def get_file_ownership_map(
        self, node_ids: Iterable[int] | None = None
    ) -> dict[int, list[str]]:
        return _persistence.get_file_ownership_map(self._conn, node_ids)

    @_synchronized
    def get_github_bind_attempt(self, goal_id: int) -> dict | None:
        return _persistence.get_github_bind_attempt(self._conn, goal_id)

    @_synchronized
    def set_github_bind_attempt(
        self,
        goal_id: int,
        marker: str,
        issue_url: str | None,
        created_at: str,
    ) -> None:
        _persistence.set_github_bind_attempt(self._conn, goal_id, marker, issue_url, created_at)

    @_synchronized
    def clear_github_bind_attempt(self, goal_id: int) -> None:
        _persistence.clear_github_bind_attempt(self._conn, goal_id)

    # ── Goal-claim fencing ───────────────────────────────────────────────────

    @_synchronized
    def claim_goal(self, goal_id: int, run_id: str, *, now: str, pid: int | None = None) -> bool:
        owner_pid = os.getpid() if pid is None else pid
        node = self.get_node(goal_id)
        if node is None:
            raise ValueError(f"node {goal_id} not found")
        if node.kind != NodeKind.GOAL:
            raise ValueError(
                f"node {goal_id} has kind={node.kind.value}; only goal nodes can be claimed"
            )
        return _goal_claims.claim_goal_row(self._conn, goal_id, run_id, now, pid=owner_pid)

    @_synchronized
    def claim_or_reclaim_goal(
        self, goal_id: int, owner: str, pid: int | None = None, *, now: str
    ) -> bool:
        owner_pid = os.getpid() if pid is None else pid
        node = self.get_node(goal_id)
        if node is None:
            raise ValueError(f"node {goal_id} not found")
        if node.kind != NodeKind.GOAL:
            raise ValueError(
                f"node {goal_id} has kind={node.kind.value}; only goal nodes can be claimed"
            )
        return _goal_claims.claim_or_reclaim_goal(self._conn, goal_id, owner, owner_pid, now=now)

    @_synchronized
    def set_goal_pid(self, goal_id: int, run_id: str, pid: int) -> None:
        _goal_claims.set_goal_claim_pid(self._conn, goal_id, run_id, pid)

    @_synchronized
    def release_goal(self, goal_id: int, run_id: str) -> bool:
        return _goal_claims.release_goal_row(self._conn, goal_id, run_id)

    @_synchronized
    def claim_ancestor_goal(self, node_id: int, run_id: str, pid: int, *, now: str) -> int | None:
        goal_id = _goal_claims.find_ancestor_goal_id(self._conn, node_id)
        if goal_id is None:
            return None
        if not _goal_claims.claim_goal_row(self._conn, goal_id, run_id, now, pid=pid):
            existing = _goal_claims.get_goal_claim(self._conn, goal_id)
            if existing is None or existing["pid"] != pid:
                raise ValueError(f"ancestor goal {goal_id} is already claimed; dispatch refused")
        return goal_id

    @_synchronized
    def try_reclaim_goal(self, goal_id: int, *, now: str) -> bool:
        return _goal_claims.try_reclaim_goal(self._conn, goal_id, now=now)

    @_synchronized
    def ancestor_goal_claimed_by_other(
        self, node_id: int, *, caller_run_id: str | None = None
    ) -> dict | None:
        return _goal_claims.ancestor_goal_claimed_by_other(
            self._conn, node_id, caller_run_id=caller_run_id
        )

    @_synchronized
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
