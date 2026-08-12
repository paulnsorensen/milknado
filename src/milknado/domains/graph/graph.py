from __future__ import annotations

import logging
import os
import sqlite3
import traceback
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from milknado.domains.common import (
    GraphExecutionSnapshot,
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
    _rebalance,
    _status,
)
from milknado.domains.graph._analytics_facade import _AnalyticsFacade, _synchronized
from milknado.domains.graph._edge_facade import _EdgeFacade
from milknado.domains.graph._pipeline import StatusPipeline, _PluginAsMiddleware

if TYPE_CHECKING:
    from milknado.domains.common import PluginHook
    from milknado.domains.graph.rebalance import ReapOutcome, ReapTarget, RebalanceState
_logger = logging.getLogger(__name__)


class MikadoGraph(_AnalyticsFacade, _EdgeFacade):
    """Thin facade over the graph slice's free-function modules.

    Public methods delegate to `_reads`/`_creation`/`_status`/`_persistence`/
    `_mutations`; status transitions run through a `StatusPipeline` so plugins
    observe before/after each change. Rich docstrings live on the free functions.
    """

    def __init__(self, db_path: Path, plugins: Sequence[PluginHook] = ()) -> None:
        self._lock = RLock()
        self._db_path = db_path
        self._closed = False
        self._close_stack: list[str] | None = None
        self._raw_conn = self._open(db_path)
        self._pipeline = StatusPipeline([_PluginAsMiddleware(p) for p in plugins])
        self._dispatch_exclusions: set[int] = set()

    @classmethod
    def open_snapshot(cls, db_path: Path) -> MikadoGraph:
        """Open a migrated in-memory copy without writing to the project database."""
        uri = f"{db_path.resolve().as_uri()}?mode=ro"
        source = sqlite3.connect(uri, uri=True, check_same_thread=False)
        snapshot = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            if source.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise sqlite3.DatabaseError(f"database failed PRAGMA quick_check: {db_path}")
            source.backup(snapshot)
            snapshot.row_factory = sqlite3.Row
            snapshot.execute("PRAGMA foreign_keys=ON")
            _persistence.create_tables(snapshot)
            _persistence.migrate(snapshot)
        except Exception:
            snapshot.close()
            raise
        finally:
            source.close()
        graph = cls.__new__(cls)
        graph._lock = RLock()
        graph._db_path = db_path
        graph._closed = False
        graph._close_stack = None
        graph._raw_conn = snapshot
        graph._pipeline = StatusPipeline([])
        graph._dispatch_exclusions = set()
        return graph

    # ── Connection lifecycle (self-heal chokepoint) ──────────────────────────

    @staticmethod
    def _quarantine(db_path: Path) -> list[Path]:
        """Move db + -wal/-shm sidecars aside to *.corrupt-<ts>; return moved paths."""
        # Microsecond resolution: two quarantines of the same db within one
        # second must not collide and silently overwrite forensic evidence.
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")
        moved: list[Path] = []
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{db_path}{suffix}")
            if candidate.exists():
                target = Path(f"{candidate}.corrupt-{ts}")
                candidate.rename(target)
                moved.append(target)
        return moved

    def _open(self, db_path: Path) -> sqlite3.Connection:
        """Open the database, quarantining a corrupt file and healing schema drift."""
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
        except sqlite3.DatabaseError:
            quick_check = "error"
        if quick_check != "ok":
            # Quarantine BEFORE closing: close() on the last connection makes
            # sqlite delete the -wal/-shm sidecars, destroying the very
            # evidence (and data) the quarantine exists to preserve.
            moved = self._quarantine(db_path)
            try:
                conn.close()
            except sqlite3.Error:
                # Best-effort close: the db was already renamed aside by the
                # quarantine, so this handle to the moved file is abandoned
                # either way — a close failure changes nothing.
                pass
            _logger.warning(
                "database failed PRAGMA quick_check (%s); quarantined %s to %s; recreating fresh",
                quick_check,
                db_path,
                [str(p) for p in moved],
            )
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # Explicit so the concurrent-writer wait window (detached runner + server
        # both writing the same db) is documented, not implicit in connect()'s default.
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            _persistence.create_tables(conn)
            _persistence.migrate(conn)
        except Exception:
            # Never leak the handle on a schema/migration failure — close
            # best-effort, then re-raise the original error unchanged.
            try:
                conn.close()
            except sqlite3.Error:
                pass
            raise
        return conn

    def _heal_conn(self, reason: str) -> None:
        """Reopen an unexpectedly-closed connection, logging both stacks.

        The recorded close stack names who closed it; the heal stack names who
        tripped over it — the pair is the audit trail for the lifetime bug.
        """
        heal_stack = traceback.format_stack()
        _logger.warning(
            "graph connection reopened unexpectedly (%s); db=%s"
            "\nclose stack:\n%s\nheal stack:\n%s",
            reason,
            self._db_path,
            "".join(self._close_stack) if self._close_stack else "(no close() recorded)",
            "".join(heal_stack),
        )
        self._raw_conn = self._open(self._db_path)
        self._closed = False

    @property
    def _conn(self) -> sqlite3.Connection:
        """Health-checked connection access: reopen on unexpected close.

        Healing happens on the NEXT access; a statement already in flight on the
        dead connection raises its original error — no transparent retry.
        """
        raw = self._raw_conn
        if self._closed or raw is None:
            self._heal_conn("closed" if self._closed else "missing")
        else:
            try:
                raw.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                self._heal_conn("connection died without close()")
        if self._raw_conn is None:
            raise RuntimeError("graph connection healing did not produce a connection")
        return self._raw_conn

    # ── Creation ─────────────────────────────────────────────────────────────

    @_synchronized
    def add_node(
        self, description: str, parent_id: int | None = None, spec: NodeSpec | None = None
    ) -> MikadoNode:
        return _creation.add_node(self._conn, description, parent_id, spec or NodeSpec())

    @_synchronized
    def set_batch_metadata(self, node_id: int, oversized: bool, batch_index: int | None) -> None:
        _creation.set_batch_metadata(self._conn, node_id, oversized, batch_index)

    @_synchronized
    def set_parent_id(self, node_id: int, parent_id: int | None) -> None:
        _creation.set_parent_id(self._conn, node_id, parent_id)

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
    def archive_subtree(self, node_id: int) -> int:
        """Soft-hide an all-DONE subtree; returns nodes archived. Fail-loud on live work."""
        return _mutations.archive_subtree(self._conn, node_id)

    @_synchronized
    def unarchive_subtree(self, node_id: int) -> int:
        """Cascade-restore an archived subtree; refuses under an archived ancestor."""
        return _mutations.unarchive_subtree(self._conn, node_id)

    @_synchronized
    def set_todo_status(self, node_id: int, target: NodeStatus) -> bool:
        """Apply one todo status request after complete preflight."""
        return _status.set_todo_status(self._pipeline, self._conn, node_id, target)

    @_synchronized
    def set_subtree_status(self, root_id: int, target: NodeStatus) -> int:
        """Set status across root_id's live (non-archived) subtree, children first."""
        return _status.set_subtree_status(self._pipeline, self._conn, root_id, target)

    @_synchronized
    def reconcile_completed_goals(self) -> int:
        """Reconcile completed goal children through the state machine."""
        return _status.reconcile_completed_goals(self._pipeline, self._conn)

    @_synchronized
    def set_dispatch_exclusions(self, node_ids: set[int]) -> None:
        self._dispatch_exclusions = set(node_ids)

    @_synchronized
    def dispatch_exclusions(self) -> set[int]:
        return set(self._dispatch_exclusions)

    @_synchronized
    def move_node(self, node_id: int, new_parent_id: int | None) -> None:
        _mutations.reparent(self._conn, node_id, new_parent_id)

    @_synchronized
    def rebalance(self, *, sweep: bool = True, restructure: bool = True) -> RebalanceState:
        """Apply graph-only rebalance passes in one database transaction."""
        return _rebalance.rebalance_graph(self._conn, sweep=sweep, restructure=restructure)

    @_synchronized
    def get_reap_targets(self) -> tuple[ReapTarget, ...]:
        """Return archived worktrees that have no running run."""
        return _rebalance.get_reap_targets(self._conn)

    @_synchronized
    def reap_archived(
        self, processor: Callable[[ReapTarget], ReapOutcome]
    ) -> tuple[ReapOutcome, ...]:
        """Process eligible worktrees while archive eligibility is write-fenced."""
        return _rebalance.reap_archived(self._conn, processor)

    @_synchronized
    def count_archived(self) -> int:
        return _rebalance.count_archived(self._conn)

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
    def get_all_nodes(self, *, include_archived: bool = False) -> list[MikadoNode]:
        return _reads.get_all_nodes(self._conn, include_archived=include_archived)

    @_synchronized
    def get_children(self, node_id: int, *, include_archived: bool = False) -> list[MikadoNode]:
        return _reads.get_children(self._conn, node_id, include_archived=include_archived)

    @_synchronized
    def get_children_map(self, *, include_archived: bool = False) -> dict[int, list[MikadoNode]]:
        return _reads.get_children_map(self._conn, include_archived=include_archived)

    @_synchronized
    def get_leaves(self, *, include_archived: bool = False) -> list[MikadoNode]:
        return _reads.get_leaves(self._conn, include_archived=include_archived)

    @_synchronized
    def get_ready_nodes(
        self,
        *,
        kind: NodeKind | None = None,
        flavor: str | None = None,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[MikadoNode]:
        return _reads.get_ready_nodes(
            self._conn, kind=kind, flavor=flavor, limit=limit, include_archived=include_archived
        )

    @_synchronized
    def get_node_summaries(
        self,
        *,
        status: NodeStatus | None = None,
        kind: NodeKind | None = None,
        flavor: str | None = None,
        page: tuple[int, int] = (100, 0),
        include_archived: bool = False,
    ) -> list[dict[str, int | str]]:
        return _reads.get_node_summaries(
            self._conn,
            status=status,
            kind=kind,
            flavor=flavor,
            page=page,
            include_archived=include_archived,
        )

    @_synchronized
    def get_root(self, *, include_archived: bool = False) -> MikadoNode | None:
        return _reads.get_root(self._conn, include_archived=include_archived)

    @_synchronized
    def get_roots(self, *, include_archived: bool = False) -> list[MikadoNode]:
        return _reads.get_roots(self._conn, include_archived=include_archived)

    @_synchronized
    def get_execution_snapshot(self, node_ids: list[int]) -> GraphExecutionSnapshot:
        """Return one lock-held snapshot of graph facts for execution policy."""
        self._conn.execute("SAVEPOINT execution_snapshot")
        try:
            ready = _reads.get_ready_nodes(self._conn)
            running = tuple(
                node.id
                for node in _reads.get_all_nodes(self._conn)
                if node.status is NodeStatus.RUNNING
            )
            ready_ids = tuple(node.id for node in ready)
            conflicts = _persistence.check_parallel_safety(self._conn, [*running, *ready_ids])
            return GraphExecutionSnapshot(
                root=_reads.get_root(self._conn),
                nodes=tuple(_reads.get_nodes(self._conn, node_ids)),
                ready_node_ids=ready_ids,
                running_node_ids=running,
                parallel_conflicts=tuple(
                    (left_id, right_id, tuple(paths)) for left_id, right_id, paths in conflicts
                ),
            )
        finally:
            self._conn.execute("RELEASE SAVEPOINT execution_snapshot")

    # ── Status transitions (pipeline-driven) ─────────────────────────────────

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
    def mark_terminal(
        self,
        node_id: int,
        run_id: str,
        status: NodeStatus,
        *,
        preserve_recovery: bool = False,
    ) -> bool:
        return _status.mark_terminal(
            self._pipeline,
            self._conn,
            node_id,
            run_id,
            status,
            preserve_recovery=preserve_recovery,
        )

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
            f"UPDATE nodes SET {column} = ? WHERE id = ?",
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
    def deposit_review_verdict(
        self, run_id: str, verdict: str, findings: str, created_at: str
    ) -> int:
        return _persistence.deposit_review_verdict(
            self._conn, run_id, verdict, findings, created_at
        )

    @_synchronized
    def latest_run_message(self, run_id: str, role: str) -> str | None:
        return _persistence.latest_run_message(self._conn, run_id, role)

    @_synchronized
    def insert_node_review(
        self,
        node_id: int,
        round_number: int,
        verdict: str,
        findings: str,
        created_at: str,
    ) -> None:
        _persistence.insert_node_review(
            self._conn, node_id, round_number, verdict, findings, created_at
        )

    @_synchronized
    def node_reviews_for_node(self, node_id: int) -> list[dict]:
        return _persistence.node_reviews_for_node(self._conn, node_id)

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
        # Already closed: no-op WITHOUT touching _close_stack — the recorded
        # stack names the close that actually closed, and a redundant close
        # must not clobber the forensic pair the heal warning relies on.
        if self._closed:
            return
        # Record the calling stack BEFORE closing: an unexpected reopen later
        # logs this stack next to the heal stack, naming the culprit pair.
        self._close_stack = traceback.format_stack()
        conn = self._raw_conn
        self._closed = True
        self._raw_conn = None
        if conn is None:
            return
        # Fold the WAL into the main .db file so the documented
        # `git add -f .milknado/milknado.db` persistence step captures every
        # committed write. A non-last-connection close does no implicit
        # checkpoint, so without this an MCP tool call or detached run can leave
        # its tail in the -wal sidecar and lose it on container reclaim.
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            _logger.warning("WAL checkpoint on close failed", exc_info=True)
        finally:
            conn.close()
