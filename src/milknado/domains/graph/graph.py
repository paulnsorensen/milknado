from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common import (
    VALID_CHILD_KINDS,
    MikadoEdge,
    MikadoNode,
    NodeKind,
    NodeSpec,
    NodeStatus,
    RunResult,
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
    find_ancestor_goal_id,
    finish_run,
    get_file_ownership,
    get_file_ownership_map,
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

    def add_node(
        self, description: str, parent_id: int | None = None, spec: NodeSpec | None = None
    ) -> MikadoNode:
        spec = spec or NodeSpec()
        if not isinstance(spec.kind, NodeKind):
            raise ValueError(f"invalid kind: {spec.kind!r}")
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._validate_parent(parent_id, spec.kind)
            self._validate_prereqs(spec.prereqs, parent_id)
            flavor = self._validate_flavor(spec.kind, spec.flavor, spec.flavor_registry)
            node_id = self._insert_node(description, parent_id, spec, flavor)
            if parent_id is not None:
                self._conn.execute(
                    "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
                    (parent_id, node_id),
                )
            for prereq_id in spec.prereqs:
                if self._creates_cycle(node_id, prereq_id):
                    raise ValueError(f"Edge {node_id}->{prereq_id} would create a cycle")
            self._conn.executemany(
                "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
                [(node_id, prereq_id) for prereq_id in spec.prereqs],
            )
        except Exception:
            self._conn.rollback()
            raise
        self._conn.commit()
        _logger.debug("node %d created: %r", node_id, description[:80])
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return row_to_node(row)

    def _insert_node(
        self, description: str, parent_id: int | None, spec: NodeSpec, flavor: str | None
    ) -> int:
        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO nodes "
            "(description, status, parent_id, created_at, oversized, batch_index, "
            "kind, flavor, wiki_ref, github_ref, artifact_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                description,
                NodeStatus.PENDING.value,
                parent_id,
                now,
                int(spec.oversized),
                spec.batch_index,
                spec.kind.value,
                flavor,
                spec.wiki_ref,
                spec.github_ref,
                spec.artifact_path,
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("add_node INSERT did not return lastrowid")
        return cur.lastrowid

    def _validate_parent(self, parent_id: int | None, kind: NodeKind) -> None:
        if parent_id is None:
            return
        parent = self.get_node(parent_id)
        if parent is None:
            raise ValueError(f"parent_id {parent_id} not found")
        if kind not in VALID_CHILD_KINDS.get(parent.kind, set()):
            raise InvalidContainment(parent.kind, kind)

    def _validate_prereqs(self, prereqs: Sequence[int], parent_id: int | None) -> None:
        if len(prereqs) != len(set(prereqs)):
            raise ValueError("prereqs contains duplicate node ids")
        for prereq_id in prereqs:
            if self.get_node(prereq_id) is None:
                raise ValueError(f"prereq {prereq_id} not found")
            if prereq_id == parent_id:
                raise ValueError(f"prereq {prereq_id} duplicates parent_id {parent_id}")

    def _validate_flavor(
        self, kind: NodeKind, flavor: str | None, flavor_registry: frozenset[str]
    ) -> str | None:
        # flavor validation: TASK defaults to IMPLEMENT, non-TASK must be None
        if kind != NodeKind.TASK and flavor is not None:
            raise ValueError(f"flavor must be None for kind={kind.value}; got {flavor!r}")
        flavor_value = (flavor or DEFAULT_FLAVOR) if kind == NodeKind.TASK else None
        if flavor_value is not None and flavor_value not in flavor_registry:
            raise ValueError(
                f"unknown flavor {flavor_value!r}; valid flavors: {sorted(flavor_registry)}"
            )
        return flavor_value

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
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if self._creates_cycle(parent_id, child_id):
                raise ValueError(f"Edge {parent_id}->{child_id} would create a cycle")
            self._conn.execute(
                "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
                (parent_id, child_id),
            )
        except Exception:
            self._conn.rollback()
            raise
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
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        node = row_to_node(row)
        if node.kind == NodeKind.GOAL:
            claim = get_goal_claim(self._conn, node_id)
            if claim is not None:
                return replace(node, goal_run_id=claim["run_id"])
        return node

    def get_nodes(self, node_ids: Iterable[int]) -> list[MikadoNode]:
        """Load nodes in input order with duplicate IDs preserved."""
        ids = list(node_ids)
        if not ids:
            return []
        unique_ids = list(dict.fromkeys(ids))
        rows = []
        for start in range(0, len(unique_ids), 500):
            chunk = unique_ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                self._conn.execute(
                    f"SELECT * FROM nodes WHERE id IN ({placeholders})",
                    chunk,  # noqa: S608
                ).fetchall()
            )
        nodes = {row["id"]: row_to_node(row) for row in rows}
        goal_ids = [node_id for node_id, node in nodes.items() if node.kind == NodeKind.GOAL]
        if goal_ids:
            placeholders = ",".join("?" for _ in goal_ids)
            claims = self._conn.execute(
                f"SELECT goal_id, run_id FROM goal_claims WHERE goal_id IN ({placeholders})",  # noqa: S608
                goal_ids,
            ).fetchall()
            for goal_id, run_id in claims:
                nodes[goal_id] = replace(nodes[goal_id], goal_run_id=run_id)
        return [nodes[node_id] for node_id in ids if node_id in nodes]

    def latest_results_for_nodes(self, node_ids: Iterable[int]) -> dict[int, str]:
        """Return each requested node's newest deposited result body."""
        ids = list(dict.fromkeys(node_ids))
        results: dict[int, str] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                "SELECT node_id, body FROM ("
                "SELECT r.node_id, m.body, ROW_NUMBER() OVER ("
                "PARTITION BY r.node_id ORDER BY m.created_at DESC, m.seq DESC"
                ") AS ordinal FROM runs r JOIN run_messages m ON m.run_id = r.run_id "
                f"WHERE m.role = 'result' AND r.node_id IN ({placeholders})"  # noqa: S608
                ") WHERE ordinal = 1",
                chunk,
            ).fetchall()
            results.update((row["node_id"], row["body"]) for row in rows)
        return results

    def find_node_by_wiki_ref(self, wiki_ref: str) -> MikadoNode | None:
        """Return the node carrying this deterministic wiki key, or None.

        The round-trip lookup the importer/exporter use: wiki_ref is computed
        from git-resident frontmatter, identical across dbs, so it is the stable
        join between a wiki goal file and its milknado node.
        """
        row = self._conn.execute("SELECT * FROM nodes WHERE wiki_ref = ?", (wiki_ref,)).fetchone()
        return row_to_node(row) if row else None

    def find_node_by_github_ref(self, github_ref: str) -> MikadoNode | None:
        """Return the node carrying this GitHub Projects node id, or None.

        The stable join between a GitHub project/item and its milknado node:
        github_ref is the opaque PVT/PVTI id, identical across dbs, so it is the
        idempotency key the github importer/binder round-trip on.
        """
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE github_ref = ?", (github_ref,)
        ).fetchone()
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

    def get_file_ownership_map(self) -> dict[int, list[str]]:
        """Map node_id -> owned file paths, reusing the persistence id-scan.

        Lets callers materialise ownership for an entire subtree without issuing
        one get_file_ownership query per node (the N+1 pattern).
        """
        return get_file_ownership_map(self._conn)

    def get_leaves(self) -> list[MikadoNode]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT parent_id FROM edges)"
        ).fetchall()
        return [row_to_node(r) for r in rows]

    def get_ready_nodes(
        self,
        *,
        kind: NodeKind | None = None,
        flavor: str | None = None,
        limit: int = 100,
    ) -> list[MikadoNode]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        filters = ["n.status = ?", "EXISTS (SELECT 1 FROM edges i WHERE i.child_id = n.id)"]
        params: list[object] = [NodeStatus.PENDING.value]
        if kind is not None:
            filters.append("n.kind = ?")
            params.append(kind.value)
        if flavor is not None:
            filters.append("n.flavor = ?")
            params.append(flavor)
        params.append(limit)
        rows = self._conn.execute(
            "SELECT n.* FROM nodes n WHERE "
            + " AND ".join(filters)
            + " AND NOT EXISTS (SELECT 1 FROM edges e JOIN nodes c ON c.id = e.child_id "
            "WHERE e.parent_id = n.id AND c.status != 'done') ORDER BY n.id LIMIT ?",
            params,
        ).fetchall()
        return [row_to_node(row) for row in rows]

    def get_node_summaries(
        self,
        *,
        status: NodeStatus | None = None,
        kind: NodeKind | None = None,
        flavor: str | None = None,
        page: tuple[int, int] = (100, 0),
    ) -> list[dict[str, int | str]]:
        limit, offset = page
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        filters, params = [], []
        for column, value in (
            ("status", status.value if status else None),
            ("kind", kind.value if kind else None),
            ("flavor", flavor),
        ):
            if value is not None:
                filters.append(f"{column} = ?")
                params.append(value)
        where = f" WHERE {' AND '.join(filters)}" if filters else ""
        params.extend((limit, offset))
        rows = self._conn.execute(
            f"SELECT id, status, description FROM nodes{where} ORDER BY id LIMIT ? OFFSET ?",  # noqa: S608
            params,
        ).fetchall()
        return [dict(row) for row in rows]

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
        ready = self.get_ready_nodes(kind=kind, limit=1)
        return ready[0] if ready else None

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

    def set_github_ref(self, node_id: int, github_ref: str) -> None:
        """Attach a GitHub Projects node id to an already-created node (used by
        bind to record the project/item ids on the roadmap and goal nodes)."""
        cur = self._conn.execute(
            "UPDATE nodes SET github_ref = ? WHERE id = ?",
            (github_ref, node_id),
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

    def claim_ancestor_goal_for_dispatch(
        self, node_id: int, run_id: str, *, now: str | None = None
    ) -> None:
        """Enforce and claim the ancestor-goal subtree fence before dispatch.

        Raises ValueError if an ancestor goal is claimed by a different live
        coordinator (a different pid). Same-process dispatch (same pid) is
        exempt: the current process is the coordinator that holds the claim,
        so sibling tasks may proceed. No-op claim when there is no ancestor
        goal. Pass ``now`` to share one timestamp with a paired node claim;
        it defaults to the current UTC time.
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
        node_id itself could not be claimed (already claimed / wrong status).
        """
        self.claim_ancestor_goal_for_dispatch(node_id, run_id, now=now)
        if not self.claim_node(node_id, run_id, now=now):
            current = self.get_node(node_id)
            status = current.status.value if current is not None else "gone"
            raise ValueError(
                f"node {node_id} is already {status}; set status back to pending to retry"
            )

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

    def finish_run(self, run_id: str, result: RunResult) -> None:
        """UPDATE a run to a terminal status (replaces the terminal sidecar write)."""
        finish_run(self._conn, run_id, result)

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

    def claim_or_reclaim_goal(self, goal_id: int, owner: str, pid: int, *, now: str) -> bool:
        """Atomically acquire a goal claim, replacing a provably dead owner."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("SELECT kind FROM nodes WHERE id = ?", (goal_id,)).fetchone()
            if row is None:
                raise ValueError(f"node {goal_id} not found")
            if row["kind"] != NodeKind.GOAL.value:
                raise ValueError(
                    f"node {goal_id} has kind={row['kind']}; only goal nodes can be claimed"
                )
            claim = get_goal_claim(self._conn, goal_id)
            if claim is not None and claim["run_id"] != owner:
                prior_pid = claim["pid"]
                if prior_pid is None or pid_alive(prior_pid):
                    self._conn.rollback()
                    return False
                self._conn.execute("DELETE FROM goal_claims WHERE goal_id = ?", (goal_id,))
            self._conn.execute(
                "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(goal_id) DO UPDATE SET pid=excluded.pid, "
                "claimed_at=excluded.claimed_at "
                "WHERE goal_claims.run_id=excluded.run_id",
                (goal_id, owner, pid, now),
            )
        except Exception:
            self._conn.rollback()
            raise
        self._conn.commit()
        return True

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
