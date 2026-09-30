"""Recover durable loop workers before releasing a node or run."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from milknado.domains.common import ObservationKey, WorkerIdentity
from milknado.domains.dispatch.ports import (
    ProcessTerminationPort,
    WorkerCleanupResult,
    WorkerRecoveryPort,
)
from milknado.domains.graph import (
    NodeWorkers,
    RunWorkers,
    UnassociatedWorkers,
    WorkerEvidenceStore,
    WorkerRecord,
)

_logger = logging.getLogger(__name__)
_RECOVERY_TIMEOUT_SECONDS = 8.0


@dataclass(frozen=True, slots=True)
class ReapRequest:
    selection: NodeWorkers | RunWorkers | UnassociatedWorkers
    deadline: float | None = None
    db_path: Path | None = None


def _identity(record: WorkerRecord) -> WorkerIdentity:
    return WorkerIdentity(record.invocation_id, record.pid, record.pgid, record.start_token)


def _prepare_worker(
    evidence: WorkerEvidenceStore, process: ProcessTerminationPort, record: WorkerRecord
) -> WorkerRecord | None:
    if record.observation_owner is not None:
        _logger.error(
            "worker recovery unresolved: invocation_id=%s interrupted observation=%s sequence=%s",
            record.invocation_id, record.observation_owner, record.observation_seq,
        )
        return None
    key = ObservationKey(
        record.invocation_id, "supervisor", record.snapshot_seq + 1,
        -1, record.pid, record.start_token,
    )
    try:
        evidence.begin(key)
        observed = process.observe_worker(_identity(record))
        evidence.commit(key, observed)
        refreshed = evidence.get(record.invocation_id)
    except Exception:
        _logger.exception(
            "worker recovery observation unresolved: invocation_id=%s", record.invocation_id
        )
        return None
    if refreshed is None or refreshed.observation_owner is not None or refreshed.ended_at is not None:
        _logger.error("worker recovery evidence changed: invocation_id=%s", record.invocation_id)
        return None
    return refreshed


def _stop_worker(
    process: ProcessTerminationPort, record: WorkerRecord, deadline: float
) -> WorkerCleanupResult:
    if time.monotonic() >= deadline:
        return WorkerCleanupResult(False, ("worker recovery deadline expired",))
    try:
        return process.terminate_worker(_identity(record), record.descendants, deadline)
    except Exception as exc:
        _logger.exception("worker recovery cleanup failed: invocation_id=%s", record.invocation_id)
        return WorkerCleanupResult(False, (f"cleanup error: {exc}",))


def _run_stop(
    process: ProcessTerminationPort,
    record: WorkerRecord,
    deadline: float,
    outcomes: Queue[tuple[str, WorkerCleanupResult]],
) -> None:
    outcomes.put((record.invocation_id, _stop_worker(process, record, deadline)))


def _stop_workers(
    process: ProcessTerminationPort, records: tuple[WorkerRecord, ...], deadline: float
) -> dict[str, WorkerCleanupResult]:
    outcomes: Queue[tuple[str, WorkerCleanupResult]] = Queue()
    for record in records:
        Thread(target=_run_stop, args=(process, record, deadline, outcomes), daemon=True).start()
    completed: dict[str, WorkerCleanupResult] = {}
    while len(completed) < len(records) and (remaining := deadline - time.monotonic()) > 0:
        try:
            invocation_id, result = outcomes.get(timeout=remaining)
        except Empty:
            break
        completed[invocation_id] = result
    return completed

def reap_orphaned_workers(
    graph: WorkerRecoveryPort, process: ProcessTerminationPort, request: ReapRequest
) -> bool:
    """Close records only after all covered targets have confirmed exit."""
    deadline = request.deadline or time.monotonic() + _RECOVERY_TIMEOUT_SECONDS
    path = graph.db_path if request.db_path is None else request.db_path
    try:
        with WorkerEvidenceStore(path, deadline=deadline) as evidence:
            records = evidence.live_workers(request.selection)
            recoverable: list[WorkerRecord] = []
            complete = True
            for record in records:
                if isinstance(request.selection, UnassociatedWorkers):
                    state = process.supervisor_state(
                        record.supervisor_pid, record.supervisor_start_token
                    )
                    if state == "live":
                        continue
                    if state != "gone":
                        complete = False
                        _logger.error(
                            "worker owner unresolved: invocation_id=%s supervisor_pid=%s state=%s",
                            record.invocation_id, record.supervisor_pid, state,
                        )
                        continue
                recoverable.append(record)
            prepared = tuple(
                refreshed
                for record in recoverable
                if time.monotonic() < deadline
                and (refreshed := _prepare_worker(evidence, process, record)) is not None
            )
            complete = complete and len(prepared) == len(recoverable)
            if not prepared:
                return complete
            if time.monotonic() >= deadline:
                _logger.error("worker recovery deadline expired before cleanup: selection=%s", request.selection)
                return False
            outcomes = _stop_workers(process, prepared, deadline)
            for record in prepared:
                result = outcomes.get(record.invocation_id, WorkerCleanupResult(
                    False, ("worker recovery deadline expired",)
                ))
                if not result.covered_exited or time.monotonic() >= deadline:
                    complete = False
                    _logger.error(
                        "worker recovery unresolved: invocation_id=%s graph_run_id=%s node_id=%s identities=%s",
                        record.invocation_id, record.graph_run_id, record.node_id, result.unresolved,
                    )
                    continue
                try:
                    evidence.end(record.invocation_id, record.snapshot_seq)
                except Exception:
                    complete = False
                    _logger.exception(
                        "worker recovery closure unresolved: invocation_id=%s", record.invocation_id
                    )
            return complete
    except Exception:
        _logger.exception("worker recovery evidence unavailable: selection=%s", request.selection)
        return False
