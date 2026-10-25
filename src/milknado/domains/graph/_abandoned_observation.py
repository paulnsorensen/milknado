"""Release an observation abandoned by its dead supervisor."""

from __future__ import annotations

import sqlite3

from milknado.domains.graph._worker_persistence import WorkerRecord


def release_abandoned_observation(conn: sqlite3.Connection, record: WorkerRecord) -> None:
    """Clear the observation `record` shows only while the record's own supervisor holds it.

    The caller proves that supervisor gone first; an observation begun by any other
    observer, or one that moved since `record` was read, keeps its fence.
    """
    cur = conn.execute(
        """UPDATE run_workers SET observation_owner = NULL, observation_seq = NULL,
        observation_generation = NULL, observation_pid = NULL, observation_token = NULL
        WHERE invocation_id = ? AND ended_at IS NULL AND observation_owner = 'supervisor'
        AND observation_seq = ? AND observation_pid = supervisor_pid
        AND observation_token = supervisor_start_token
        AND supervisor_pid = ? AND supervisor_start_token = ?""",
        (
            record.invocation_id,
            record.observation_seq,
            record.supervisor_pid,
            record.supervisor_start_token,
        ),
    )
    conn.commit()
    if cur.rowcount != 1:
        raise RuntimeError("worker observation fence lost")
