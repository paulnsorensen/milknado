"""Rebalance engine: sweep archivable subtrees and group orphan tasks.

Pure domain over a raw sqlite connection (db_first slice convention): these
functions never commit — the app layer owns the write transaction so a dry-run
is the same selection logic with no writes at all. Selection (read-only) is
split from mutation so the app layer can plan a dry-run without opening a
write transaction: `find_archivable_roots`/`orphan_candidates`/`structural_report`
never write; `sweep_archivable`/`group_orphans` stamp and move.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection

from milknado.domains.graph._persistence import children_id_map, create_tables, migrate

INBOX_DESCRIPTION = "Inbox (milknado-managed)"

# Structural report thresholds (spec minor decisions).
MIN_CHAIN_DEPTH = 4
LOPSIDED_TASK_THRESHOLD = 20


def open_db(db_path: Path) -> Connection:
    """Open the milknado db with MikadoGraph's conventions (graph.py `_open`).

    WAL journal + foreign keys + busy_timeout=5000 — the concurrent-writer
    wait window the live coordinator + detached runner environment needs —
    then create/migrate the schema.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    create_tables(conn)
    migrate(conn)
    return conn


def _subtree_ids(children: dict[int, list[int]], root_id: int) -> list[int]:
    """root_id plus every descendant via the canonical children map."""
    ids: list[int] = []
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        ids.append(node_id)
        stack.extend(children.get(node_id, ()))
    return ids


def _descendant_ids(conn: Connection, root_id: int) -> list[int]:
    """root_id plus every descendant via the edges table (recursive CTE)."""
    rows = conn.execute(
        """
        WITH RECURSIVE subtree(id) AS (
            SELECT ?
            UNION
            SELECT e.child_id
            FROM edges e
            JOIN subtree s ON e.parent_id = s.id
        )
        SELECT id FROM subtree
        """,
        (root_id,),
    ).fetchall()
    return [row[0] for row in rows]


def is_claimed(conn: Connection, root_id: int) -> bool:
    """True when any goal inside root_id's subtree holds a goal_claims row.

    Presence is the test: claims are deleted on release/terminal transition,
    so a row means an active (or at worst reclaimable-by-owner) fence — either
    way the sweep must not archive out from under it.
    """
    ids = _descendant_ids(conn, root_id)
    placeholders = ",".join("?" for _ in ids)
    row = conn.execute(
        f"SELECT 1 FROM goal_claims WHERE goal_id IN ({placeholders}) LIMIT 1",  # noqa: S608
        ids,
    ).fetchone()
    return row is not None


def _all_done_map(status: dict[int, str], children: dict[int, list[int]]) -> dict[int, bool]:
    """node_id -> True when the node and its whole subtree are DONE.

    Iterative post-order over the forest (the canonical edges-derived children
    map), memoized so shared work between roots is computed once.
    """
    memo: dict[int, bool] = {}
    for root in status:
        if root in memo:
            continue
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if node in memo:
                continue
            kids = [kid for kid in children.get(node, ()) if kid in status]
            if not processed:
                stack.append((node, True))
                stack.extend((kid, False) for kid in kids)
            else:
                memo[node] = status[node] == "done" and all(memo[kid] for kid in kids)
    return memo


def find_archivable_roots(conn: Connection) -> list[int]:
    """Read-only: ids of every maximal all-DONE, unarchived, unclaimed subtree.

    Maximal means the node's own subtree is all-DONE and its parent's is not
    (or it has no parent), so a finished goal nested under a live roadmap
    qualifies just as a top-level finished root does. Claimed subtrees are
    never archivable. Used by both the real sweep and the dry-run plan.
    """
    rows = conn.execute("SELECT id, status, archived_at FROM nodes").fetchall()
    status = {row[0]: row[1] for row in rows}
    archived_at = {row[0]: row[2] for row in rows}
    children = children_id_map(conn)
    parent_of = {child: parent for parent, kids in children.items() for child in kids}
    all_done = _all_done_map(status, children)
    roots: list[int] = []
    for node_id in sorted(status):
        if archived_at[node_id] is not None or not all_done[node_id]:
            continue
        parent = parent_of.get(node_id)
        if parent is not None and all_done.get(parent, False):
            continue  # not maximal — the parent is itself an all-DONE candidate
        if is_claimed(conn, node_id):
            continue
        roots.append(node_id)
    return roots


def sweep_archivable(conn: Connection) -> list[int]:
    """Archive every maximal all-DONE subtree; return the archived root ids.

    Stamps `archived_at` on every unarchived node in each selected subtree;
    already-archived members keep their original stamp (idempotent re-sweep).
    Never commits — the caller owns the transaction.
    """
    roots = find_archivable_roots(conn)
    if not roots:
        return []
    children = children_id_map(conn)
    now = datetime.now(UTC).isoformat()
    for root_id in roots:
        ids = _subtree_ids(children, root_id)
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"UPDATE nodes SET archived_at = ? WHERE id IN ({placeholders}) "  # noqa: S608
            "AND archived_at IS NULL",
            [now, *ids],
        )
    return roots


def reap_eligible_ids(conn: Connection, *, sweep_pending: bool) -> set[int]:
    """Ids whose worktrees reap may target: archived nodes only.

    Archive gates destruction. For a dry-run plan (`sweep_pending=True`) the
    set additionally includes every node in a subtree the sweep pass is about
    to archive, so the plan reflects post-sweep state without writing.
    """
    ids = {row[0] for row in conn.execute("SELECT id FROM nodes WHERE archived_at IS NOT NULL")}
    if sweep_pending:
        children = children_id_map(conn)
        for root_id in find_archivable_roots(conn):
            ids.update(_subtree_ids(children, root_id))
    return ids


def find_inbox_id(conn: Connection) -> int | None:
    """Id of the NON-archived root GOAL carrying the reserved Inbox description.

    An archived Inbox is shelved history and must never attract live orphans
    (acceptance #5: nothing is added under an archived parent), so the
    find-or-create path ignores it and creates a fresh Inbox instead.
    """
    row = conn.execute(
        "SELECT id FROM nodes WHERE parent_id IS NULL AND kind = 'goal' "
        "AND description = ? AND archived_at IS NULL",
        (INBOX_DESCRIPTION,),
    ).fetchone()
    return row[0] if row is not None else None


def _find_or_create_inbox(conn: Connection) -> int:
    inbox_id = find_inbox_id(conn)
    if inbox_id is not None:
        return inbox_id
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        "INSERT INTO nodes (description, status, parent_id, kind, created_at) "
        "VALUES (?, 'pending', NULL, 'goal', ?)",
        (INBOX_DESCRIPTION, now),
    )
    inbox_id = cur.lastrowid
    assert inbox_id is not None  # lastrowid is set after a successful INSERT
    return inbox_id


def orphan_candidates(conn: Connection) -> list[int]:
    """Read-only: ids of orphan TASKs that would attach under the Inbox.

    An orphan is a non-archived TASK whose parent is missing (NULL or
    dangling) or archived. Claimed subtrees are respected and left in place.
    """
    rows = conn.execute(
        """
        SELECT n.id FROM nodes n
        WHERE n.kind = 'task'
          AND n.archived_at IS NULL
          AND (
              n.parent_id IS NULL
              OR NOT EXISTS (SELECT 1 FROM nodes p WHERE p.id = n.parent_id)
              OR EXISTS (
                  SELECT 1 FROM nodes p
                  WHERE p.id = n.parent_id AND p.archived_at IS NOT NULL
              )
          )
        ORDER BY n.id
        """,
    ).fetchall()
    return [node_id for (node_id,) in rows if not is_claimed(conn, node_id)]


def group_orphans(conn: Connection) -> int:
    """Attach root-level orphan TASKs under the find-or-create Inbox GOAL.

    The Inbox is found-or-created only when at least one orphan will actually
    attach — a zero-orphan pass creates nothing. Reparenting updates both
    `nodes.parent_id` and the `edges` row. Returns the number of attached
    tasks. Never commits.
    """
    candidates = orphan_candidates(conn)
    if not candidates:
        return 0
    inbox_id = _find_or_create_inbox(conn)
    for node_id in candidates:
        conn.execute("UPDATE nodes SET parent_id = ? WHERE id = ?", (inbox_id, node_id))
        conn.execute("DELETE FROM edges WHERE child_id = ?", (node_id,))
        conn.execute(
            "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
            (inbox_id, node_id),
        )
    return len(candidates)


@dataclass(frozen=True)
class StructureReport:
    """Report-only structural findings: deep chains and lopsided goals.

    `chains` holds each maximal single-child GOAL chain of depth >= 4 as the
    tuple of goal ids from chain head to tail; `lopsided` holds goal ids with
    >= 20 direct TASK children. Findings are never mutated, only reported.
    """

    chains: tuple[tuple[int, ...], ...] = ()
    lopsided: tuple[int, ...] = ()


def structural_report(conn: Connection, *, exclude_ids: Iterable[int] = ()) -> StructureReport:
    """Find deep single-child GOAL chains and lopsided GOALs. Never mutates.

    Only the visible (unarchived) forest is considered: archived nodes are
    shelved history, not structure the user still needs to act on.
    `exclude_ids` treats additional nodes as already shelved without writing
    — the dry-run passes the about-to-be-archived set so its findings match
    the post-sweep forest the real run reports on.
    """
    rows = conn.execute("SELECT id, kind FROM nodes WHERE archived_at IS NULL").fetchall()
    excluded = set(exclude_ids)
    kind_of = {row[0]: row[1] for row in rows if row[0] not in excluded}
    children = {
        parent: [child for child in kids if child in kind_of]
        for parent, kids in children_id_map(conn).items()
        if parent in kind_of
    }

    def sole_goal_child(node_id: int) -> int | None:
        kids = children.get(node_id, ())
        if len(kids) == 1 and kind_of.get(kids[0]) == "goal":
            return kids[0]
        return None

    mid_chain = {
        child
        for node_id, kind in kind_of.items()
        if kind == "goal" and (child := sole_goal_child(node_id)) is not None
    }
    chains: list[tuple[int, ...]] = []
    for goal in sorted(node_id for node_id, kind in kind_of.items() if kind == "goal"):
        if goal in mid_chain:
            continue  # not a chain head — the maximal chain starts higher
        path = [goal]
        while (nxt := sole_goal_child(path[-1])) is not None and nxt not in path:
            path.append(nxt)
        if len(path) >= MIN_CHAIN_DEPTH:
            chains.append(tuple(path))
    lopsided = tuple(
        sorted(
            node_id
            for node_id, kind in kind_of.items()
            if kind == "goal"
            and sum(1 for child in children.get(node_id, ()) if kind_of[child] == "task")
            >= LOPSIDED_TASK_THRESHOLD
        )
    )
    return StructureReport(chains=tuple(chains), lopsided=lopsided)


@dataclass(frozen=True)
class RebalanceReport:
    """Outcome of one rebalance pass, ordered sweep -> restructure -> reap."""

    archived_roots: tuple[int, ...] = ()
    inbox_id: int | None = None
    moved_tasks: int = 0
    chains: tuple[tuple[int, ...], ...] = ()
    lopsided: tuple[int, ...] = ()
    reaped: tuple[str, ...] = ()
    branches_deleted: tuple[str, ...] = ()
    preserved: tuple[str, ...] = ()
    branches_kept: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())


def render_report(report: RebalanceReport, dry_run: bool = False) -> str:
    """Render the report as fixed-order lines with equal-width counters.

    Order: sweep -> restructure (moves, then report-only structural findings)
    -> reap. Dry-run prefixes every line with ``[dry-run] Would `` and mutates
    nothing; counters are zero-padded to a common width so the two modes align.
    """
    counts = (
        len(report.archived_roots),
        report.moved_tasks,
        len(report.chains),
        len(report.lopsided),
        len(report.reaped),
        len(report.branches_deleted),
        len(report.preserved),
        len(report.branches_kept),
    )
    width = max(1, max(len(str(count)) for count in counts))
    archived, moved, chains, lopsided, reaped, branches, preserved, _kept = (
        f"{count:0{width}d}" for count in counts
    )
    prefix = "[dry-run] Would " if dry_run else ""
    roots = ", ".join(str(root) for root in report.archived_roots) or "none"
    if report.inbox_id is not None:
        inbox = f"id {report.inbox_id}"
    elif dry_run and report.moved_tasks > 0:
        inbox = "new"  # the real run find-or-creates the Inbox at apply time
    else:
        inbox = "no Inbox"
    chain_roots = ", ".join(str(chain[0]) for chain in report.chains) or "none"
    lopsided_ids = ", ".join(str(goal) for goal in report.lopsided) or "none"
    reap_line = (
        f"{prefix}Reap: removed {reaped} worktree(s), deleted {branches} branch(es), "
        f"preserved {preserved}"
    )
    if report.branches_kept:
        kept = f"{len(report.branches_kept):0{width}d}"
        kept_names = ", ".join(report.branches_kept)
        reap_line += f", kept {kept} branch(es) (unmerged: {kept_names})"
    lines = [
        f"{prefix}Sweep: archived {archived} subtree(s) (roots: {roots})",
        f"{prefix}Restructure: moved {moved} orphan task(s) under Inbox ({inbox})",
        f"{prefix}Restructure: found {chains} deep chain(s) (roots: {chain_roots})",
        f"{prefix}Restructure: found {lopsided} lopsided goal(s) (ids: {lopsided_ids})",
        reap_line,
    ]
    lines.extend(f"{prefix}Note: {note}" for note in report.notes)
    return "\n".join(lines)
