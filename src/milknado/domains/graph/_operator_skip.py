"""Operator skip: move a failed node to DONE in one graph transaction."""

from __future__ import annotations

import sqlite3

import milknado.domains.graph._reads as _reads
from milknado.domains.common import NodeStatus
from milknado.domains.graph._pipeline import StatusPipeline
from milknado.domains.graph._status import apply_todo_status


def skip_failed(pipeline: StatusPipeline, conn: sqlite3.Connection, node_id: int) -> None:
    """Apply FAILED -> PENDING -> RUNNING -> DONE atomically.

    The scheduler never sees the intermediate PENDING state, so it cannot
    admit and dispatch the node while the operator skip is in flight. A
    rejected step (done verification, archive, admission) rolls back all steps.
    """
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        status = _reads.node_status(conn, node_id)
        if status is None:
            raise ValueError(f"Node {node_id} not found")
        if status is not NodeStatus.FAILED:
            raise ValueError(f"node {node_id} is {status.value}; only a failed node skips")
        _ = apply_todo_status(pipeline, conn, node_id, NodeStatus.PENDING)
        _ = apply_todo_status(pipeline, conn, node_id, NodeStatus.DONE)
