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
        worker.identity.invocation_id,
        "supervisor",
        record.snapshot_seq + 1,
        0,
        parent.pid,
        parent.create_time(),
    )
    state = identity_state(worker.identity.pid, worker.identity.start_token)
    if state == "gone" and worker.process.poll() is not None:
        return record.snapshot_seq
    if state != "live":
        raise RuntimeError("worker identity unresolved before observation")
    evidence.begin_worker_observation(key)
    descendants = observe_descendants(worker.identity)
    state = identity_state(worker.identity.pid, worker.identity.start_token)
    if state != "live" and not (state == "gone" and worker.process.poll() is not None):
        raise RuntimeError("worker identity unresolved during observation")
    evidence.commit_worker_observation(key, descendants)
    return key.sequence
