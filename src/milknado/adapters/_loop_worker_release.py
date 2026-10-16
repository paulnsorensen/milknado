"""Record graph attempt evidence around the release of a gated loop worker."""

from __future__ import annotations

from pathlib import Path

from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_gate import StartEvidence, WorkerProcess


def release_worker(worker: WorkerProcess, owner: WorkerOwner, db_path: Path) -> None:
    run_id = owner.graph_run_id
    if run_id is None:
        worker.release()
        return
    graph = MikadoGraph(db_path)
    try:
        _release_linked_worker(worker, graph, run_id)
    finally:
        graph.close()


def _release_linked_worker(worker: WorkerProcess, graph: MikadoGraph, run_id: str) -> None:
    row = graph.runs.get(run_id)
    if row is None:
        raise RuntimeError("graph run missing before worker release")
    attempt_id = row["attempt_id"]
    if attempt_id is None:
        worker.release()
        return
    attempt = graph.task_attempts.get(attempt_id)
    if attempt is None:
        raise RuntimeError("linked task attempt missing before worker release")
    node = graph.get_node(row["node_id"])
    if node is None or node.run_id is None:
        raise RuntimeError("task attempt owner fence lost before worker release")
    owner_run_id = node.run_id
    invocation_id = worker.identity.invocation_id
    attempts = graph.task_attempts
    if not attempts.bind_worker_invocation((attempt_id, owner_run_id, run_id, invocation_id)):
        raise RuntimeError("task attempt invocation fence lost before worker release")
    worker.release()
    if attempt.max_node_attempts > 1 and not attempts.record_release(
        attempt_id, owner_run_id, invocation_id
    ):
        raise RuntimeError("task attempt owner fence lost after worker release")
    if worker.start_evidence is StartEvidence.EXECUTED:
        recorded = attempts.confirm_execution(attempt_id, owner_run_id, invocation_id)
    elif worker.start_evidence is StartEvidence.STARTUP_FAILED:
        recorded = attempts.record_startup_failure(attempt_id, owner_run_id, invocation_id)
    else:
        return
    if not recorded:
        raise RuntimeError("task attempt start evidence fence lost")
