"""Goal-claim persistence for MikadoGraph (coordinator subtree fencing).

Split out of _persistence.py to shrink that module toward its line-size
ceiling; see _analytics_facade.py for the sibling split-off-for-size precedent.
"""

from __future__ import annotations

import sqlite3
from typing import TypedDict, cast

from milknado.domains.common import NodeKind, pid_alive
from milknado.domains.graph._sqlite_rows import fetchone

GoalClaim = TypedDict(  # noqa: UP013
    "GoalClaim",
    {"goal_id": int, "run_id": str, "pid": int | None, "claimed_at": str},
)


def _field(row: object, name: str) -> object:
    if isinstance(row, sqlite3.Row):
        return cast(object, row[name])
    return cast(dict[str, object], row)[name]


def claim_goal_row(
    conn: sqlite3.Connection, goal_id: int, run_id: str, now: str, *, pid: int | None
) -> bool:
    """Acquire a goal claim with a non-null PID in one conditional write.

    The write lock covers both the liveness decision and the replacement CAS.
    A caller without a PID cannot create a claim: a PID-less owner is unsafe to
    reclaim and therefore cannot be a successful owner.
    """
    if pid is None:
        return False

    _ = conn.execute("BEGIN IMMEDIATE")
    try:
        inserted = conn.execute(
            "INSERT OR IGNORE INTO goal_claims (goal_id, run_id, pid, claimed_at) "
            + "VALUES (?, ?, ?, ?)",
            (goal_id, run_id, pid, now),
        ).rowcount
        if inserted == 1:
            conn.commit()
            return True
        claim = fetchone(conn, "SELECT run_id, pid FROM goal_claims WHERE goal_id = ?", (goal_id,))
        if claim is None:
            conn.rollback()
            return False
        claim_run_id = cast(str, _field(claim, "run_id"))
        claim_pid = cast(int | None, _field(claim, "pid"))
        if claim_run_id == run_id:
            updated = 0
        elif claim_pid is not None and not pid_alive(claim_pid):
            updated = conn.execute(
                "UPDATE goal_claims SET run_id = ?, pid = ?, claimed_at = ? "
                + "WHERE goal_id = ? AND run_id = ? AND pid = ?",
                (run_id, pid, now, goal_id, claim_run_id, claim_pid),
            ).rowcount
        else:
            updated = 0
        if updated == 1:
            conn.commit()
            return True
        conn.rollback()
        return False
    except Exception:
        conn.rollback()
        raise


def release_goal_row(conn: sqlite3.Connection, goal_id: int, run_id: str) -> bool:
    """Delete a goal claim gated on the owning run_id. Returns True iff deleted."""
    cur = conn.execute(
        "DELETE FROM goal_claims WHERE goal_id = ? AND run_id = ?",
        (goal_id, run_id),
    )
    conn.commit()
    return cur.rowcount == 1


def release_goal_row_unconditional(conn: sqlite3.Connection, goal_id: int) -> None:
    """Delete any goal claim for goal_id without a run_id gate.

    Used on terminal transitions (DONE/FAILED) for goal nodes: the goal is
    complete so the fence owner is moot — the claim must be cleared regardless
    of which run held it.
    """
    _ = conn.execute("DELETE FROM goal_claims WHERE goal_id = ?", (goal_id,))
    conn.commit()


def get_goal_claim(conn: sqlite3.Connection, goal_id: int) -> GoalClaim | None:
    """Return the goal claim row as a dict, or None if unclaimed."""
    row = fetchone(
        conn,
        "SELECT goal_id, run_id, pid, claimed_at FROM goal_claims WHERE goal_id = ?",
        (goal_id,),
    )
    if row is None:
        return None
    return {
        "goal_id": cast(int, _field(row, "goal_id")),
        "run_id": cast(str, _field(row, "run_id")),
        "pid": cast(int | None, _field(row, "pid")),
        "claimed_at": cast(str, _field(row, "claimed_at")),
    }


def ancestor_goal_claim(
    conn: sqlite3.Connection, node_id: int, caller_run_id: str | None
) -> GoalClaim | None:
    """Walk node's ancestor chain; return the first foreign goal claim."""
    current_id: int | None = node_id
    while current_id is not None:
        row = fetchone(conn, "SELECT parent_id, kind FROM nodes WHERE id = ?", (current_id,))
        if row is None:
            break
        parent_id = cast(int | None, _field(row, "parent_id"))
        kind = cast(str, _field(row, "kind"))
        if kind == "goal":
            claim = get_goal_claim(conn, current_id)
            if claim is not None and claim["run_id"] != caller_run_id:
                return claim
        current_id = parent_id
    return None


def find_ancestor_goal_id(conn: sqlite3.Connection, node_id: int) -> int | None:
    """Return the id of the closest ancestor (or self) with kind='goal', or None."""
    current_id: int | None = node_id
    while current_id is not None:
        row = fetchone(conn, "SELECT parent_id, kind FROM nodes WHERE id = ?", (current_id,))
        if row is None:
            break
        if cast(str, _field(row, "kind")) == "goal":
            return current_id
        current_id = cast(int | None, _field(row, "parent_id"))
    return None


def release_goal_claim_on_terminal(conn: sqlite3.Connection, node_id: int) -> None:
    """If node_id is a GOAL, unconditionally delete its claim on terminal transition.

    Called after mark_done / mark_failed / mark_terminal so the completed
    goal's claim does not permanently block re-dispatch under that goal.
    """
    row = fetchone(conn, "SELECT kind FROM nodes WHERE id = ?", (node_id,))
    if row is not None and cast(str, _field(row, "kind")) == NodeKind.GOAL.value:
        release_goal_row_unconditional(conn, node_id)


def claim_or_reclaim_goal(
    conn: sqlite3.Connection, goal_id: int, owner: str, pid: int | None, *, now: str
) -> bool:
    """Acquire a goal claim, replacing only a specifically observed dead owner."""
    if pid is None:
        return False
    _ = conn.execute("BEGIN IMMEDIATE")
    try:
        row = fetchone(conn, "SELECT kind FROM nodes WHERE id = ?", (goal_id,))
        if row is None:
            raise ValueError(f"node {goal_id} not found")
        kind = cast(str, _field(row, "kind"))
        if kind != NodeKind.GOAL.value:
            raise ValueError(f"node {goal_id} has kind={kind}; only goal nodes can be claimed")
        inserted = conn.execute(
            "INSERT OR IGNORE INTO goal_claims (goal_id, run_id, pid, claimed_at) "
            + "VALUES (?, ?, ?, ?)",
            (goal_id, owner, pid, now),
        ).rowcount
        if inserted == 1:
            conn.commit()
            return True
        claim = fetchone(conn, "SELECT run_id, pid FROM goal_claims WHERE goal_id = ?", (goal_id,))
        if claim is None:
            conn.rollback()
            return False
        claim_run_id = cast(str, _field(claim, "run_id"))
        claim_pid = cast(int | None, _field(claim, "pid"))
        if claim_run_id == owner:
            updated = 0
        elif claim_pid is not None and not pid_alive(claim_pid):
            updated = conn.execute(
                "UPDATE goal_claims SET run_id = ?, pid = ?, claimed_at = ? "
                + "WHERE goal_id = ? AND run_id = ? AND pid = ?",
                (owner, pid, now, goal_id, claim_run_id, claim_pid),
            ).rowcount
        else:
            updated = 0
        if updated == 1:
            conn.commit()
            return True
        conn.rollback()
        return False
    except Exception:
        conn.rollback()
        raise


def try_reclaim_goal(conn: sqlite3.Connection, goal_id: int, *, now: str) -> bool:
    """Free a claim only when its non-null PID is provably dead."""
    _ = now
    _ = conn.execute("BEGIN IMMEDIATE")
    try:
        claim = get_goal_claim(conn, goal_id)
        if claim is None or claim["pid"] is None or pid_alive(claim["pid"]):
            conn.rollback()
            return False
        deleted = conn.execute(
            "DELETE FROM goal_claims WHERE goal_id = ? AND run_id = ? AND pid = ?",
            (goal_id, claim["run_id"], claim["pid"]),
        ).rowcount
        conn.commit()
        return deleted == 1
    except Exception:
        conn.rollback()
        raise


def ancestor_goal_claimed_by_other(
    conn: sqlite3.Connection, node_id: int, *, caller_run_id: str | None = None
) -> GoalClaim | None:
    """Return a blocking goal claim dict when an ancestor has another live owner."""
    claim = ancestor_goal_claim(conn, node_id, caller_run_id)
    if claim is None:
        return None
    pid = claim["pid"]
    if pid is None:
        return claim
    if not pid_alive(pid):
        _ = release_goal_row(conn, claim["goal_id"], claim["run_id"])
        return None
    return claim
