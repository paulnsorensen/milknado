"""Goal-claim persistence for MikadoGraph (coordinator subtree fencing).

Split out of _persistence.py to shrink that module toward its line-size
ceiling; see _analytics_facade.py for the sibling split-off-for-size precedent.
"""

from __future__ import annotations

import sqlite3

from milknado.domains.common import NodeKind, pid_alive


def claim_goal_row(
    conn: sqlite3.Connection, goal_id: int, run_id: str, now: str, *, pid: int | None = None
) -> bool:
    """INSERT a goal claim if none exists; returns True iff this caller won.

    INSERT OR IGNORE is the mutual-exclusion point: a single conditional write
    that either inserts (this caller wins) or finds the row already there (loses).
    Pass pid to write it atomically in the same INSERT, eliminating the NULL-pid
    window that would otherwise allow a concurrent reclaim before set_goal_claim_pid.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?)",
        (goal_id, run_id, pid, now),
    )
    conn.commit()
    return cur.rowcount == 1


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
    conn.execute("DELETE FROM goal_claims WHERE goal_id = ?", (goal_id,))
    conn.commit()


def set_goal_claim_pid(conn: sqlite3.Connection, goal_id: int, run_id: str, pid: int) -> None:
    """Record the coordinator pid on a goal claim, gated on the owning run_id."""
    conn.execute(
        "UPDATE goal_claims SET pid = ? WHERE goal_id = ? AND run_id = ?",
        (pid, goal_id, run_id),
    )
    conn.commit()


def get_goal_claim(conn: sqlite3.Connection, goal_id: int) -> dict | None:
    """Return the goal claim row as a dict, or None if unclaimed."""
    row = conn.execute(
        "SELECT goal_id, run_id, pid, claimed_at FROM goal_claims WHERE goal_id = ?",
        (goal_id,),
    ).fetchone()
    if row is None:
        return None
    return {"goal_id": row[0], "run_id": row[1], "pid": row[2], "claimed_at": row[3]}


def ancestor_goal_claim(
    conn: sqlite3.Connection, node_id: int, caller_run_id: str | None
) -> dict | None:
    """Walk node's ancestor chain; return the first goal_claim owned by a DIFFERENT run.

    Returns None (allowed) when:
    - no ancestor goal exists
    - the ancestor goal is unclaimed
    - the claim belongs to the same caller_run_id (own coordinator)
    Returns the claim dict (blocked) when a live OR dead-but-not-yet-reclaimed foreign
    claim is found; the caller is responsible for checking pid-liveness.
    """
    # Walk parent chain collecting each ancestor's kind + claim status.
    current_id: int | None = node_id
    while current_id is not None:
        row = conn.execute(
            "SELECT parent_id, kind FROM nodes WHERE id = ?", (current_id,)
        ).fetchone()
        if row is None:
            break
        parent_id, kind = row["parent_id"], row["kind"]
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
        row = conn.execute(
            "SELECT parent_id, kind FROM nodes WHERE id = ?", (current_id,)
        ).fetchone()
        if row is None:
            break
        if row["kind"] == "goal":
            return current_id
        current_id = row["parent_id"]
    return None


def release_goal_claim_on_terminal(conn: sqlite3.Connection, node_id: int) -> None:
    """If node_id is a GOAL, unconditionally delete its claim on terminal transition.

    Called after mark_done / mark_failed / mark_terminal so the completed
    goal's claim does not permanently block re-dispatch under that goal.
    """
    row = conn.execute("SELECT kind FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row is not None and row["kind"] == NodeKind.GOAL.value:
        release_goal_row_unconditional(conn, node_id)


def claim_or_reclaim_goal(
    conn: sqlite3.Connection, goal_id: int, owner: str, pid: int, *, now: str
) -> bool:
    """Atomically acquire a goal claim, replacing a provably dead owner."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT kind FROM nodes WHERE id = ?", (goal_id,)).fetchone()
        if row is None:
            raise ValueError(f"node {goal_id} not found")
        if row["kind"] != NodeKind.GOAL.value:
            raise ValueError(
                f"node {goal_id} has kind={row['kind']}; only goal nodes can be claimed"
            )
        claim = get_goal_claim(conn, goal_id)
        if claim is not None and claim["run_id"] != owner:
            prior_pid = claim["pid"]
            if prior_pid is None or pid_alive(prior_pid):
                conn.rollback()
                return False
            conn.execute("DELETE FROM goal_claims WHERE goal_id = ?", (goal_id,))
        conn.execute(
            "INSERT INTO goal_claims (goal_id, run_id, pid, claimed_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(goal_id) DO UPDATE SET pid=excluded.pid, "
            "claimed_at=excluded.claimed_at "
            "WHERE goal_claims.run_id=excluded.run_id",
            (goal_id, owner, pid, now),
        )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return True


def try_reclaim_goal(conn: sqlite3.Connection, goal_id: int, *, now: str) -> bool:  # noqa: ARG001
    """Free a goal claim whose owner pid is provably dead.

    Returns True iff a dead owner was released so a new claimant can win.
    A live or pid-unknown owner is left intact.
    """
    claim = get_goal_claim(conn, goal_id)
    if claim is None:
        return False
    pid = claim["pid"]
    if pid is None or pid_alive(pid):
        return False
    # Dead pid: delete the stale claim so a fresh claimant can win.
    return release_goal_row(conn, goal_id, claim["run_id"])


def ancestor_goal_claimed_by_other(
    conn: sqlite3.Connection, node_id: int, *, caller_run_id: str | None = None
) -> dict | None:
    """Return a blocking goal claim dict if an ancestor goal is owned by a different run.

    Performs pid-liveness reclaim in-line: a dead foreign claimant is freed
    and the check returns None (allowed), mirroring claim_node's dead-owner reclaim.
    Returns None when dispatch is allowed; returns the claim dict when blocked.
    """
    claim = ancestor_goal_claim(conn, node_id, caller_run_id)
    if claim is None:
        return None
    pid = claim["pid"]
    if pid is None or not pid_alive(pid):
        # NULL or dead pid: reclaim and allow dispatch.
        release_goal_row(conn, claim["goal_id"], claim["run_id"])
        return None
    return claim
