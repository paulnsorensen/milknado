"""Persist supervisor observations before sampling worker descendants."""

from __future__ import annotations

import subprocess

import psutil

from milknado.domains.common import ObservationKey, identity_state
from milknado.loop._process_contract import WorkerEvidence
from milknado.loop._process_gate import WorkerProcess
from milknado.loop._process_identity import observe_descendants

REAP_WAIT_SECONDS = 5.0


def _reaped(process: subprocess.Popen[str] | subprocess.Popen[bytes]) -> bool:
    """Return whether the exited worker has a return code.

    poll() returns None while another thread holds the waitpid lock inside wait(),
    so a gone worker needs a bounded wait() instead.
    """
    if process.poll() is not None:
        return True
    try:
        _ = process.wait(timeout=REAP_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        return False
    return True


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
    if state == "gone" and _reaped(worker.process):
        return record.snapshot_seq
    if state != "live":
        raise RuntimeError("worker identity unresolved before observation")
    evidence.begin_worker_observation(key)
    descendants = observe_descendants(worker.identity)
    state = identity_state(worker.identity.pid, worker.identity.start_token)
    if state != "live" and not (state == "gone" and _reaped(worker.process)):
        raise RuntimeError("worker identity unresolved during observation")
    evidence.commit_worker_observation(key, descendants)
    return key.sequence
