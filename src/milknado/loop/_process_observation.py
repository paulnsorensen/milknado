"""Persist supervisor observations before sampling worker descendants."""

from __future__ import annotations

import psutil

from milknado.domains.common import ObservationKey
from milknado.loop._process_contract import WorkerEvidence
from milknado.loop._process_gate import WorkerProcess
from milknado.loop._process_identity import identity_state, observe_descendants


def snapshot(worker: WorkerProcess, evidence: WorkerEvidence) -> int:
    record = evidence.get_worker(worker.identity.invocation_id)
    if record is None or record.observation_owner is not None:
        raise RuntimeError("worker evidence unavailable")
    parent = psutil.Process()
    key = ObservationKey(
        worker.identity.invocation_id, "supervisor", record.snapshot_seq + 1,
        0, parent.pid, parent.create_time(),
    )
    if identity_state(worker.identity.pid, worker.identity.start_token) != "live":
        raise RuntimeError("worker exited before observation")
    evidence.begin_worker_observation(key)
    descendants = observe_descendants(worker.identity)
    if identity_state(worker.identity.pid, worker.identity.start_token) != "live":
        raise RuntimeError("worker exited during observation")
    evidence.commit_worker_observation(key, descendants)
    return key.sequence
