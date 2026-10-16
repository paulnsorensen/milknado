"""Dead-owner reclaim for a RUNNING node."""

from __future__ import annotations

import sqlite3
from typing import cast

import milknado.domains.graph._owner_release as _owner_release
import milknado.domains.graph._reads as _reads
from milknado.domains.common import NodeStatus, pid_alive
from milknado.domains.graph._pipeline import StatusPipeline
from milknado.domains.graph._sqlite_rows import as_tuple as _values
from milknado.domains.graph._sqlite_rows import fetchone


def try_reclaim(
    pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int, *, now: str
) -> bool:
    """Free a RUNNING node whose owner process is provably dead.

    Reads the current owner (run_id, pid); if the pid is recorded and no longer
    alive, releases the node back to PENDING so the next dispatch can claim it
    without waiting out the stale-running timeout. A live (or pid-unknown) owner
    is left intact — a failed claim refuses the new caller instead.

    Returns True iff a dead owner was actually released, so the dispatch boundary
    can log the recovery (an operationally interesting event on the worker path).
    """
    # Kept for symmetry with claim_node/mark_terminal's now= convention;
    # pid_alive is the sole signal here.
    _ = now
    row = fetchone(conn, "SELECT status, run_id, pid FROM nodes WHERE id = ?", (node_id,))
    if row is None:
        return False
    values = _values(row)
    if NodeStatus(cast(str, values[0])) != NodeStatus.RUNNING:
        return False
    owner_run_id = cast(str | None, values[1])
    pid = cast(int | None, values[2])
    if owner_run_id is None or pid is None or pid_alive(pid):
        return False

    def mutate() -> bool:
        return _owner_release.release(conn, node_id, owner_run_id, reclaim=True)

    return pipeline.run(
        lambda nid: _reads.get_node(conn, nid),
        node_id,
        NodeStatus.RUNNING,
        NodeStatus.PENDING,
        mutate,
    )
