"""Verify coordinator turn workers before releasing stale launch fences."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import WorkerIdentity
from milknado.domains.graph import RuntimeWorkers, WorkerEvidenceStore
from milknado.loop._process_identity import identity_state, terminate_verified


@dataclass(frozen=True, slots=True)
class CoordinatorWorkerRecovery:
    db_path: Path

    def terminated(self, turn_id: str, supervisor_pid: int, supervisor_start_token: float) -> bool:
        if identity_state(supervisor_pid, supervisor_start_token) not in {"gone", "mismatch"}:
            return False
        with WorkerEvidenceStore(self.db_path) as store:
            workers = store.live_workers(RuntimeWorkers(turn_id))
            for worker in workers:
                if (worker.supervisor_pid, worker.supervisor_start_token) != (
                    supervisor_pid,
                    supervisor_start_token,
                ):
                    return False
                identity = WorkerIdentity(
                    worker.invocation_id, worker.pid, worker.pgid, worker.start_token
                )
                if terminate_verified(identity, worker.descendants, time.monotonic() + 3):
                    return False
                try:
                    store.end(worker.invocation_id, worker.snapshot_seq, worker.helper_generation)
                except RuntimeError:
                    current = store.get(worker.invocation_id)
                    if current is None or current.ended_at is None:
                        return False
        return identity_state(supervisor_pid, supervisor_start_token) in {"gone", "mismatch"}
