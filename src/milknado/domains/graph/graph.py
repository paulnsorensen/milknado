from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common import (
    VALID_CHILD_KINDS,
    MikadoEdge,
    MikadoNode,
    NodeKind,
    NodeStatus,
    pid_alive,
)
from milknado.domains.common.errors import InvalidContainment
from milknado.domains.common.types import BUILTIN_FLAVORS, DEFAULT_FLAVOR
from milknado.domains.graph import _transitions
from milknado.domains.graph._analytics_facade import _AnalyticsFacade
from milknado.domains.graph._mutations import (
    delete_subtree,
    reparent,
    update_node_fields,
    would_create_cycle,
)
from milknado.domains.graph._persistence import (
    ancestor_goal_claim,
    check_parallel_safety,
    children_id_map,
    claim_goal_row,
    create_tables,
    deposit_run_message,
    drop_all,
    ensure_schema,
    find_ancestor_goal_id,
    finish_run,
    get_file_ownership,
    get_goal_claim,
    get_run,
    latest_run_message,
    recent_runs,
    release_goal_row,
    release_goal_row_unconditional,
    row_to_node,
    runs_for_node,
    set_file_ownership,
    set_goal_claim_pid,
    set_pid,
    set_run_pid,
    set_worktree,
    start_run,
)

if TYPE_CHECKING:
    from milknado.domains.common import PluginHook

_logger = logging.getLogger(__name__)


class MikadoGraph(_AnalyticsFacade):
    def __init__(self, db_path: Path, plugins: Sequence[PluginHook] = ()) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # Explicit so the concurrent-writer wait window (detached runner + server
        # both writing the same db) is documented, not implicit in connect()'s default.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._plugins: tuple[PluginHook, ...] = tuple(plugins)
        create_tables(self._conn)
        ensure_schema(self._conn)

    def add_node(
        self,
        description: str,
        parent_id: int | None = None,
        *,
        oversized: bool = False,
        batch_index: int | None = None,
        kind: NodeKind = NodeKind.TASK,
        flavor: str | None = None,
        wiki_ref: str | None = None,
        artifact_path: str | None = None,
        prereqs: Sequence[int] = (),
        flavor_registry: frozenset[str] = BUILTIN_FLAVORS,
    ) -> MikadoNode:
        # Validate parent_id before inserting: the node row commits before the
        # edge is added, so a nonexistent parent would otherwise leave a
        # committed stray node when the edge FK insert fails (e.g. a stale
        # MILKNADO_NODE_ID passed by milknado_track_follow_up).
        if parent_id is not None:
            parent = self.get_node(parent_id)
            if parent is None:
                raise ValueError(f"parent_id {parent_id} not found")
            if kind not in VALID_CHILD_KINDS.get(parent.kind, set()):
                raise InvalidContainment(parent.kind, kind)
        for prereq_id in prereqs:
            if self.get_node(prereq_id) is None:
                raise ValueError(f"prereq {prereq_id} not found")
            if prereq_id == parent_id:
                raise ValueError(
                    f"prereq {prereq_id} duplicates parent_id {parent_id}; "
                    "the parent edge already covers this prerequisite"
                )
        # flavor validation: TASK defaults to IMPLEMENT, non-TASK must be None
        if kind != NodeKind.TASK and flavor is not None:
            raise ValueError(f"flavor must be None for kind={kind.value}; got {flavor!r}")
        flavor_value = (flavor or DEFAULT_FLAVOR) if kind == NodeKind.TASK else None
        if flavor_value is not None and flavor_value not in flavor_registry:
            raise ValueError(
                f"unknown flavor {flavor_value!r}; valid flavors: {sorted(flavor_registry)}"
            )
        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO nodes "
            "(description, status, parent_id, created_at, oversized, batch_index, "
            "kind, flavor, wiki_ref, artifact_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                description,
                NodeStatus.PENDING.value,
                parent_id,
                now,
                1 if oversized else 0,
                batch_index,
                kind.value,
                flavor_value,
                wiki_ref,
                artifact_path,
            ),
        )
        self._conn.commit()
        node_id = cur.lastrowid
        if node_id is None:  # pragma: no cover - defensive: plain INSERT always sets lastrowid
            raise RuntimeError("add_node INSERT did not return lastrowid")
        _logger.debug("node %d created: %r", node_id, description[:80])
        if parent_id is not None:
            self.add_edge(parent_id, node_id)
        for prereq_id in prereqs:
            self.add_edge(prereq_id, node_id)
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
            "UPDATE nodes SET parent_id = ? WHERE id = ?",
            (parent_id, node_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Node {node_id} not found")
        self._conn.commit()

    def delete_node(self, node_id: int, cascade: bool = False) -> int:
        """Delete a node, returning the count removed.

        A node with children is refused unless cascade=True, which removes the
        whole subtree atomically (single transaction). See delete_subtree.
        """
        return delete_subtree(self._conn, node_id, cascade)

    def update_node(
        self,
        node_id: int,
        description: str | None = None,
        kind: NodeKind | None = None,
        flavor: str | None = None,
        artifact_path: str | None = None,
        flavor_registry: frozenset[str] = BUILTIN_FLAVORS,
    ) -> None:
        """Update description, kind, flavor, and/or artifact_path on a node. Status stays
        governed by the state machine and is not editable here."""
        update_node_fields(
            self._conn, node_id, description, kind, flavor, artifact_path, flavor_registry
        )

    def move_node(self, node_id: int, new_parent_id: int | None) -> None:
        """Re-parent a node, rewriting its edge and parent_id with cycle checks."""
        reparent(self._conn, node_id, new_parent_id)

    def add_edge(self, parent_id: int, child_id: int) -> MikadoEdge:
        if self._creates_cycle(parent_id, child_id):
            raise ValueError(f"Edge {parent_id}->{child_id} would create a cycle")
        self._conn.execute(
            "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )
        self._conn.commit()
        return MikadoEdge(parent_id=parent_id, child_id=child_id)

    def _creates_cycle(self, parent_id: int, child_id: int) -> bool:
        return would_create_cycle(self._conn, parent_id, child_id)

    def _node_status(self, node_id: int) -> NodeStatus | None:
        row = self._conn.execute("SELECT status FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return NodeStatus(row[0]) if row else None

    def _notify_status_change(
        self, node_id: int, old_status: NodeStatus, new_status: NodeStatus
    ) -> None:
        if not self._plugins:
            return
        node = self.get_node(node_id)
        if node is None:
            return
        for plugin in self._plugins:
            try:
                plugin.on_node_status_change(node, old_status, new_status)
            except Exception:
                _logger.exception("Plugin %s raised in on_node_status_change", plugin.meta.name)

    def get_node(self, node_id: int) -> MikadoNode | None:
        from dataclasses import replace

        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        node = row_to_node(row)
        if node.kind == NodeKind.GOAL:
            claim = get_goal_claim(self._conn, node_id)
            if claim is not None:
                return replace(node, goal_run_id=claim["run_id"])
        return node

    def find_node_by_wiki_ref(self, wiki_ref: str) -> MikadoNode | None:
        """Return the node carrying this deterministic wiki key, or None.

        The round-trip lookup the importer/exporter use: wiki_ref is computed
        from git-resident frontmatter, identical across dbs, so it is the stable
        join between a wiki goal file and its milknado node.
        """
        row = self._conn.execute("SELECT * FROM nodes WHERE wiki_ref = ?", (wiki_ref,)).fetchone()
        return row_to_node(row) if row else None

    def get_all_nodes(self) -> list[MikadoNode]:
        return [row_to_node(r) for r in self._conn.execute("SELECT * FROM nodes").fetchall()]

    def get_children(self, node_id: int) -> list[MikadoNode]:
        rows = self._conn.execute(
            "SELECT n.* FROM nodes n JOIN edges e ON n.id = e.child_id WHERE e.parent_id = ?",
            (node_id,),
        ).fetchall()
        return [row_to_node(r) for r in rows]

    def get_children_map(self) -> dict[int, list[MikadoNode]]:
        """Map parent_id -> child nodes, reusing the persistence id-scan.

        Lets callers materialise an entire subtree without issuing one
        get_children query per node (the N+1 pattern).
        """
        nodes = {n.id: n for n in self.get_all_nodes()}
        mapping: dict[int, list[MikadoNode]] = {}
        for parent_id, child_ids in children_id_map(self._conn).items():
            kids = [nodes[cid] for cid in child_ids if cid in nodes]
            if kids:
                mapping[parent_id] = kids
        return mapping

    def get_leaves(self) -> list[MikadoNode]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT parent_id FROM edges)"
        ).fetchall()
        return [row_to_node(r) for r in rows]

    def get_ready_nodes(self) -> list[MikadoNode]:
        root_ids = {r.id for r in self.get_roots()}
        children_map = self.get_children_map()
        ready = []
        for node in self.get_all_nodes():
            if node.status != NodeStatus.PENDING:
                continue
            if node.id in root_ids:
                continue
            children = children_map.get(node.id, [])
            if not children or all(c.status == NodeStatus.DONE for c in children):
                ready.append(node)
        return ready

    def get_root(self) -> MikadoNode | None:
        row = self._conn.execute(
            "SELECT * FROM nodes"
            " WHERE id NOT IN (SELECT DISTINCT child_id FROM edges)"
            " ORDER BY id LIMIT 1"
        ).fetchone()
        return row_to_node(row) if row else None

    def get_roots(self) -> list[MikadoNode]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT child_id FROM edges)"
        ).fetchall()
        return [row_to_node(r) for r in rows]

    def get_next_runnable(self, kind: NodeKind | None = None) -> MikadoNode | None:
        for node in self.get_ready_nodes():
            if kind is not None and node.kind != kind:
                continue
            return node
        return None

    def _transition_status(self, node_id: int, target: NodeStatus) -> None:
        old = self._node_status(node_id)
        _transitions.transition_status(self._conn, node_id, target)
        _logger.debug("node %d: %s → %s", node_id, old.value if old else "?", target.value)
        if old is not None:
            self._notify_status_change(node_id, old, target)

    def mark_done(self, node_id: int) -> None:
        self._transition_status(node_id, NodeStatus.DONE)
        self._release_goal_claim_on_terminal(node_id)

    def complete_root(self) -> bool:
        """Auto-complete root when all non-root nodes are done.

        Returns True if root was completed.
        """
        root = self.get_root()
        if root is None or root.status != NodeStatus.PENDING:
            return False
        all_nodes = self.get_all_nodes()
        non_root = [n for n in all_nodes if n.id != root.id]
        if not all(n.status == NodeStatus.DONE for n in non_root):
            return False
        self.mark_running(root.id)
        self.mark_done(root.id)
        return True

    def mark_failed(self, node_id: int) -> None:
        old = self._node_status(node_id)
        _transitions.mark_failed(self._conn, node_id)
        if old is not None:
            self._notify_status_change(node_id, old, NodeStatus.FAILED)
        self._release_goal_claim_on_terminal(node_id)

    def mark_running(
        self,
        node_id: int,
        worktree_path: str | None = None,
        branch_name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        old = self._node_status(node_id)
        _transitions.mark_running(self._conn, node_id, worktree_path, branch_name, run_id)
        if old is not None:
            self._notify_status_change(node_id, old, NodeStatus.RUNNING)

    def set_wiki_ref(self, node_id: int, wiki_ref: str) -> None:
        """Attach a deterministic wiki key to a node (used when export files an
        orphan goal node discovered mid-run, so later round-trips locate it)."""
        cur = self._conn.execute(
            "UPDATE nodes SET wiki_ref = ? WHERE id = ?",
            (wiki_ref, node_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Node {node_id} not found")
        self._conn.commit()

    def set_run_id(self, node_id: int, run_id: str) -> None:
        cur = self._conn.execute(
            "UPDATE nodes SET run_id = ? WHERE id = ?",
            (run_id, node_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Node {node_id} not found")
        self._conn.commit()

    def claim_node(self, node_id: int, run_id: str, *, now: str) -> bool:
        """Atomically claim a claimable (pending/failed/blocked) node as RUNNING.

        The cross-process mutual-exclusion point: a single conditional UPDATE that
        either wins (returns True) or loses to a concurrent claimant (False). The
        run_id written becomes the fence guarding later ownership-gated writes.
        """
        old = self._node_status(node_id)
        claimed = _transitions.claim_node(self._conn, node_id, run_id, now)
        if claimed and old is not None:
            self._notify_status_change(node_id, old, NodeStatus.RUNNING)
        return claimed

    def try_reclaim(self, node_id: int, *, now: str) -> bool:
        """Free a RUNNING node whose owner process is provably dead.

        Reads the current owner (run_id, pid); if the pid is recorded and no longer
        alive, releases the node back to PENDING so the next dispatch can claim it
        without waiting out the stale-running timeout. A live (or pid-unknown) owner
        is left intact — a failed claim refuses the new caller instead.

        Returns True iff a dead owner was actually released, so the dispatch boundary
        can log the recovery (an operationally interesting event on the worker path).
        """
        row = self._conn.execute(
            "SELECT status, run_id, pid FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None or NodeStatus(row["status"]) != NodeStatus.RUNNING:
            return False
        owner_run_id, pid = row["run_id"], row["pid"]
        if owner_run_id is None or pid is None or pid_alive(pid):
            return False
        if _transitions.release(self._conn, node_id, owner_run_id):
            self._notify_status_change(node_id, NodeStatus.RUNNING, NodeStatus.PENDING)
            return True
        return False

    def release(self, node_id: int, run_id: str) -> bool:
        """Release a RUNNING claim back to PENDING, fenced on the run_id.

        Used by dispatch cleanup to undo a claim whose startup failed, without
        clobbering a node already re-claimed under a different run. Fenced on both
        run_id and status = 'running' (see _transitions.release), so a node that
        completed DONE between the caller's read and this write is never walked
        back to PENDING. Returns whether the release landed.
        """
        ok = _transitions.release(self._conn, node_id, run_id)
        if ok:
            self._notify_status_change(node_id, NodeStatus.RUNNING, NodeStatus.PENDING)
        return ok

    def mark_terminal(self, node_id: int, run_id: str, status: NodeStatus) -> bool:
        """Write a terminal status (DONE/FAILED) gated on the run_id fence.

        Returns False when the node was re-claimed under a new run_id — the caller
        was reclaimed and its terminal write is rejected (zero rows).
        """
        ok = _transitions.mark_terminal(self._conn, node_id, run_id, status)
        if ok:
            # The fenced UPDATE matched only because status was 'running', so the
            # source state is RUNNING by construction. Emit that instead of a
            # pre-read that could be stale under cross-process contention.
            self._notify_status_change(node_id, NodeStatus.RUNNING, status)
            self._release_goal_claim_on_terminal(node_id)
        return ok

    def set_pid(self, node_id: int, run_id: str, pid: int) -> None:
        """Record the worker pid on the node, CAS-gated on the run_id fence."""
        set_pid(self._conn, node_id, run_id, pid)

    def set_worktree(
        self, node_id: int, run_id: str, worktree_path: str, branch_name: str
    ) -> None:
        """Attach worktree/branch to an already-claimed node (no status change),
        CAS-gated on the run_id fence."""
        set_worktree(self._conn, node_id, run_id, worktree_path, branch_name)

    # ── Run repository (replaces the JSON sidecars) ──────────────────────────

    def start_run(
        self,
        run_id: str,
        node_id: int,
        log_path: str,
        started_at: str,
        timeout_seconds: int | None,
        pid: int | None = None,
    ) -> None:
        """INSERT a status='running' run row (replaces the initial sidecar write)."""
        start_run(self._conn, run_id, node_id, log_path, started_at, timeout_seconds, pid)

    def finish_run(
        self,
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
        """UPDATE a run to a terminal status (replaces the terminal sidecar write)."""
        finish_run(
            self._conn,
            run_id,
            status=status,
            exit_code=exit_code,
            timed_out=timed_out,
            ended_at=ended_at,
            error=error,
            detail=detail,
            rebased=rebased,
        )

    def set_run_pid(self, run_id: str, pid: int) -> None:
        """Record the detached runner's pid on a still-running run (fenced)."""
        set_run_pid(self._conn, run_id, pid)

    def get_run(self, run_id: str) -> dict | None:
        """Return one run row as a dict, or None."""
        return get_run(self._conn, run_id)

    def runs_for_node(
        self, node_id: int, *, terminal_only: bool = False, run_id: str | None = None
    ) -> list[dict]:
        """Return this node's runs as dicts. Caller applies fence-before-latest."""
        return runs_for_node(self._conn, node_id, terminal_only=terminal_only, run_id=run_id)

    def recent_runs(self, limit: int) -> list[dict]:
        """Return the most-recently-started runs as dicts."""
        return recent_runs(self._conn, limit)

    def deposit_run_message(self, run_id: str, role: str, body: str, created_at: str) -> int:
        """Append a run message; return its seq."""
        return deposit_run_message(self._conn, run_id, role, body, created_at)

    def latest_run_message(self, run_id: str, role: str) -> str | None:
        """Return the body of the latest message for this run+role, or None."""
        return latest_run_message(self._conn, run_id, role)

    def mark_pending(self, node_id: int) -> None:
        old = self._node_status(node_id)
        _transitions.mark_pending(self._conn, node_id)
        if old is not None:
            self._notify_status_change(node_id, old, NodeStatus.PENDING)

    def mark_blocked(self, node_id: int) -> None:
        self._transition_status(node_id, NodeStatus.BLOCKED)

    def set_file_ownership(self, node_id: int, files: list[str]) -> None:
        set_file_ownership(self._conn, node_id, files)

    def get_file_ownership(self, node_id: int) -> list[str]:
        return get_file_ownership(self._conn, node_id)

    def check_parallel_safety(self, node_ids: list[int]) -> list[tuple[int, int, list[str]]]:
        return check_parallel_safety(self._conn, node_ids)

    def drop_all(self) -> int:
        return drop_all(self._conn)

    # ── Goal-claim fencing (coordinator subtree lock) ────────────────────────

    def _release_goal_claim_on_terminal(self, node_id: int) -> None:
        """If node_id is a GOAL, unconditionally delete its claim on terminal transition.

        Called after mark_done / mark_failed / mark_terminal so the completed
        goal's claim does not permanently block re-dispatch under that goal.
        """
        row = self._conn.execute("SELECT kind FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is not None and row["kind"] == NodeKind.GOAL.value:
            release_goal_row_unconditional(self._conn, node_id)

    def claim_goal(self, goal_id: int, run_id: str, *, now: str) -> bool:
        """Claim a goal node as owned by run_id. Returns True iff this caller won.

        Uses INSERT OR IGNORE as the mutual-exclusion point across processes.
        Only GOAL kind nodes may be claimed; raises ValueError otherwise.
        """
        node = self.get_node(goal_id)
        if node is None:
            raise ValueError(f"node {goal_id} not found")
        if node.kind != NodeKind.GOAL:
            raise ValueError(
                f"node {goal_id} has kind={node.kind.value}; only goal nodes can be claimed"
            )
        return claim_goal_row(self._conn, goal_id, run_id, now)

    def release_goal(self, goal_id: int, run_id: str) -> bool:
        """Release a goal claim, fenced on run_id. Returns True iff released."""
        return release_goal_row(self._conn, goal_id, run_id)

    def claim_ancestor_goal(self, node_id: int, run_id: str, pid: int, *, now: str) -> int | None:
        """Find the closest ancestor goal and claim it with run_id + pid.

        Returns the goal_id that was claimed (or already owned by this run_id),
        or None when there is no ancestor goal node.  The pid is written
        atomically in the same INSERT to eliminate the NULL-pid race window.
        """
        goal_id = find_ancestor_goal_id(self._conn, node_id)
        if goal_id is None:
            return None
        claim_goal_row(self._conn, goal_id, run_id, now, pid=pid)
        return goal_id

    def set_goal_pid(self, goal_id: int, run_id: str, pid: int) -> None:
        """Record the coordinator pid on a goal claim (CAS on run_id)."""
        set_goal_claim_pid(self._conn, goal_id, run_id, pid)

    def try_reclaim_goal(self, goal_id: int, *, now: str) -> bool:  # noqa: ARG002
        """Free a goal claim whose owner pid is provably dead.

        Returns True iff a dead owner was released so a new claimant can win.
        A live or pid-unknown owner is left intact.
        """
        claim = get_goal_claim(self._conn, goal_id)
        if claim is None:
            return False
        pid = claim["pid"]
        if pid is None or pid_alive(pid):
            return False
        # Dead pid: delete the stale claim so a fresh claimant can win.
        return release_goal_row(self._conn, goal_id, claim["run_id"])

    def ancestor_goal_claimed_by_other(
        self, node_id: int, *, caller_run_id: str | None = None
    ) -> dict | None:
        """Return a blocking goal claim dict if an ancestor goal is owned by a different run.

        Performs pid-liveness reclaim in-line: a dead foreign claimant is freed
        and the check returns None (allowed), mirroring claim_node's dead-owner reclaim.
        Returns None when dispatch is allowed; returns the claim dict when blocked.
        """
        claim = ancestor_goal_claim(self._conn, node_id, caller_run_id)
        if claim is None:
            return None
        pid = claim["pid"]
        if pid is None or not pid_alive(pid):
            # NULL or dead pid: reclaim and allow dispatch.
            release_goal_row(self._conn, claim["goal_id"], claim["run_id"])
            return None
        return claim

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
