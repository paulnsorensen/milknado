"""SQLite-backed sweep, restructure, structural analysis, and reap fencing."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from milknado.domains.graph._persistence import children_id_map
from milknado.domains.graph.rebalance import (
    INBOX_DESCRIPTION,
    ReapOutcome,
    ReapTarget,
    RebalanceState,
    StructureReport,
)

MIN_CHAIN_DEPTH = 4
LOPSIDED_TASK_THRESHOLD = 20


def _subtree_ids(children: dict[int, list[int]], root_id: int) -> list[int]:
    """Return each node in the reachable subtree exactly once."""
    ids: list[int] = []
    visited: set[int] = set()
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        ids.append(node_id)
        stack.extend(children.get(node_id, ()))
    return ids


def _all_done_map(status: dict[int, str], children: dict[int, list[int]]) -> dict[int, bool]:
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


def _find_archivable_roots(conn: sqlite3.Connection) -> list[int]:
    """Find maximal all-DONE live subtrees using every canonical parent edge."""
    rows = conn.execute("SELECT id, status, archived_at FROM nodes").fetchall()
    status = {row[0]: row[1] for row in rows}
    archived_at = {row[0]: row[2] for row in rows}
    children = children_id_map(conn)
    parents: dict[int, set[int]] = {}
    for parent_id, child_ids in children.items():
        for child_id in child_ids:
            parents.setdefault(child_id, set()).add(parent_id)
    all_done = _all_done_map(status, children)
    return [
        node_id
        for node_id in sorted(status)
        if archived_at[node_id] is None
        and all_done[node_id]
        and not any(all_done.get(parent_id, False) for parent_id in parents.get(node_id, ()))
    ]


def _sweep_archivable(conn: sqlite3.Connection) -> list[int]:
    roots = _find_archivable_roots(conn)
    if not roots:
        return []
    children = children_id_map(conn)
    stamp = datetime.now(UTC).isoformat()
    for root_id in roots:
        ids = _subtree_ids(children, root_id)
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"UPDATE nodes SET archived_at = ? WHERE id IN ({placeholders}) "
            "AND archived_at IS NULL",
            [stamp, *ids],
        )
    return roots


def _find_inbox_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT id FROM nodes WHERE parent_id IS NULL AND kind = 'goal' "
        "AND description = ? AND archived_at IS NULL ORDER BY id LIMIT 1",
        (INBOX_DESCRIPTION,),
    ).fetchone()
    return row[0] if row is not None else None


def _orphan_candidates(conn: sqlite3.Connection) -> list[int]:
    """Return exactly live root-level tasks, ordered for deterministic grouping."""
    return [
        row[0]
        for row in conn.execute(
            "SELECT id FROM nodes WHERE kind = 'task' AND parent_id IS NULL "
            "AND archived_at IS NULL ORDER BY id"
        ).fetchall()
    ]


def _find_or_create_inbox(conn: sqlite3.Connection) -> tuple[int, bool]:
    inbox_id = _find_inbox_id(conn)
    if inbox_id is not None:
        return inbox_id, False
    cur = conn.execute(
        "INSERT INTO nodes (description, status, parent_id, kind, created_at) "
        "VALUES (?, 'pending', NULL, 'goal', ?)",
        (INBOX_DESCRIPTION, datetime.now(UTC).isoformat()),
    )
    if cur.lastrowid is None:
        raise RuntimeError("Inbox INSERT did not return lastrowid")
    return cur.lastrowid, True


def _group_orphans(conn: sqlite3.Connection) -> tuple[int, int | None, bool]:
    candidates = _orphan_candidates(conn)
    if not candidates:
        return 0, _find_inbox_id(conn), False
    inbox_id, created = _find_or_create_inbox(conn)
    conn.executemany(
        "UPDATE nodes SET parent_id = ? WHERE id = ?",
        [(inbox_id, node_id) for node_id in candidates],
    )
    conn.executemany(
        "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
        [(inbox_id, node_id) for node_id in candidates],
    )
    return len(candidates), inbox_id, created


def _visible_structure(
    conn: sqlite3.Connection, exclude_ids: Iterable[int] = ()
) -> tuple[dict[int, str], dict[int, list[int]]]:
    excluded = set(exclude_ids)
    rows = conn.execute("SELECT id, kind FROM nodes WHERE archived_at IS NULL").fetchall()
    kind_of = {row[0]: row[1] for row in rows if row[0] not in excluded}
    children = {
        parent: [child for child in kids if child in kind_of]
        for parent, kids in children_id_map(conn).items()
        if parent in kind_of
    }
    return kind_of, children


def _sole_goal_child(
    node_id: int, kind_of: dict[int, str], children: dict[int, list[int]]
) -> int | None:
    kids = children.get(node_id, ())
    if len(kids) == 1 and kind_of.get(kids[0]) == "goal":
        return kids[0]
    return None


def _deep_chains(
    kind_of: dict[int, str], children: dict[int, list[int]]
) -> tuple[tuple[int, ...], ...]:
    mid_chain = {
        child
        for node_id, kind in kind_of.items()
        if kind == "goal" and (child := _sole_goal_child(node_id, kind_of, children)) is not None
    }
    chains: list[tuple[int, ...]] = []
    for goal in sorted(node_id for node_id, kind in kind_of.items() if kind == "goal"):
        if goal in mid_chain:
            continue
        path = [goal]
        while True:
            nxt = _sole_goal_child(path[-1], kind_of, children)
            if nxt is None or nxt in path:
                break
            path.append(nxt)
        if len(path) >= MIN_CHAIN_DEPTH:
            chains.append(tuple(path))
    return tuple(chains)


def _lopsided_goals(kind_of: dict[int, str], children: dict[int, list[int]]) -> tuple[int, ...]:
    return tuple(
        sorted(
            node_id
            for node_id, kind in kind_of.items()
            if kind == "goal"
            and sum(1 for child in children.get(node_id, ()) if kind_of[child] == "task")
            >= LOPSIDED_TASK_THRESHOLD
        )
    )


def structural_report(
    conn: sqlite3.Connection, exclude_ids: Iterable[int] = ()
) -> StructureReport:
    kind_of, children = _visible_structure(conn, exclude_ids)
    return StructureReport(
        chains=_deep_chains(kind_of, children),
        lopsided=_lopsided_goals(kind_of, children),
    )


def rebalance_graph(conn: sqlite3.Connection, *, sweep: bool, restructure: bool) -> RebalanceState:
    """Apply graph-only passes atomically and return their storage-agnostic outcome."""
    archived: list[int] = []
    moved, inbox_id, inbox_created = 0, None, False
    if sweep or restructure:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if sweep:
                archived = _sweep_archivable(conn)
            if restructure:
                moved, inbox_id, inbox_created = _group_orphans(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    structure = structural_report(conn) if restructure else StructureReport()
    return RebalanceState(
        archived_roots=tuple(archived),
        inbox_id=inbox_id,
        inbox_created=inbox_created,
        moved_tasks=moved,
        chains=structure.chains,
        lopsided=structure.lopsided,
    )


def get_reap_targets(conn: sqlite3.Connection) -> tuple[ReapTarget, ...]:
    """Return archived worktrees not owned by any currently running run."""
    rows = conn.execute(
        """
        SELECT n.id, n.worktree_path, n.branch_name
        FROM nodes n
        WHERE n.archived_at IS NOT NULL
          AND n.worktree_path IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM runs r WHERE r.node_id = n.id AND r.status = 'running'
          )
        ORDER BY n.id
        """
    ).fetchall()
    return tuple(ReapTarget(row[0], row[1], row[2]) for row in rows)


def reap_archived(
    conn: sqlite3.Connection, processor: Callable[[ReapTarget], ReapOutcome]
) -> tuple[ReapOutcome, ...]:
    """Fence archive eligibility while processing every destructive Git target."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        outcomes = tuple(processor(target) for target in get_reap_targets(conn))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return outcomes


def count_archived(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived_at IS NOT NULL").fetchone()
    return int(row[0])
