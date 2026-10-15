"""Acquire a gated worker and lifeline for the shared lifecycle owner."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, TypeVar

from milknado.domains.graph import MikadoGraph
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, StartEvidence, WorkerProcess, spawn_gated
from milknado.loop._process_helper import HelperStart, stop_failed_helper
from milknado.loop._process_identity import terminate_verified_result as terminate_verified
from milknado.loop._process_observation import snapshot as _snapshot
from milknado.loop._process_registry import LaunchTicket


class _ProtectedStart(Protocol):
    ticket: LaunchTicket | None

    def start_monitor(self) -> None: ...
    def shutdown(self, deadline: float) -> bool: ...
    def cleanup(self, *, deadline: float) -> bool: ...
    def close_lifeline(self) -> None: ...


TWorker = TypeVar("TWorker", bound=_ProtectedStart)
_log = logging.getLogger(__name__)
HelperStarter = Callable[
    [WorkerProcess, ProtectionContext, HelperStart], tuple[subprocess.Popen[str], int]
]


@dataclass(slots=True)
class _StartupAcquisition:
    worker: WorkerProcess
    context: ProtectionContext
    ticket: LaunchTicket | None
    recorded: bool = False
    helper: subprocess.Popen[str] | None = None
    write_fd: int | None = None
    protected: _ProtectedStart | None = None
    monitor_started: bool = False


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


def _release_worker(
    worker: WorkerProcess, context: ProtectionContext, graph: MikadoGraph | None = None
) -> None:
    run_id = context.owner.graph_run_id
    if run_id is None:
        worker.release()
        return
    owns_graph = graph is None
    graph = graph or MikadoGraph(context.db_path)
    try:
        _release_linked_worker(worker, graph, run_id)
    finally:
        if owns_graph:
            graph.close()


def _activate_worker(
    worker: WorkerProcess, context: ProtectionContext, protected: _ProtectedStart
) -> None:
    graph = MikadoGraph(context.db_path) if context.owner.graph_run_id is not None else None
    try:
        ticket = protected.ticket
        if ticket is not None:
            if not ticket.activate(
                lambda: _release_worker(worker, context, graph), protected.shutdown
            ):
                raise RuntimeError("worker admission closed before READY")
        else:
            _release_worker(worker, context, graph)
    finally:
        if graph is not None:
            graph.close()


def spawn_protected(
    options: SpawnOptions,
    context: ProtectionContext,
    start_helper: HelperStarter,
    worker_type: Callable[[WorkerProcess, subprocess.Popen[str], int, ProtectionContext], TWorker],
) -> TWorker:
    ticket = (
        context.registry.reserve(context.owner.graph_run_id)
        if context.registry is not None
        else None
    )
    try:
        worker = spawn_gated(options)
    except Exception:
        if ticket is not None:
            ticket.close()
        raise
    if ticket is not None and not ticket.bind_pending(worker.close_gate):
        try:
            _ = worker.process.wait(timeout=3)
        finally:
            _close_pipes(worker)
            ticket.close()
        raise RuntimeError("worker admission closed during launch")
    acquired = _StartupAcquisition(worker, context, ticket)
    deadline = time.monotonic() + 8
    evidence = context.evidence.with_deadline(deadline)
    try:
        evidence.record_worker(context.owner, worker.identity)
        acquired.recorded = True
        sequence = _snapshot(worker, evidence)
        acquired.helper, acquired.write_fd = start_helper(
            worker, context, HelperStart(sequence, 0, deadline)
        )
        protected = worker_type(worker, acquired.helper, acquired.write_fd, context)
        acquired.protected = protected
        protected.ticket = ticket
        _activate_worker(worker, context, protected)
        acquired.monitor_started = True
        protected.start_monitor()
        return protected
    except Exception:
        _abort_start(acquired)
        raise


def _abort_start(acquired: _StartupAcquisition) -> None:
    worker = acquired.worker
    deadline = time.monotonic() + 3
    worker.close_gate()
    try:
        if acquired.protected is not None and acquired.monitor_started:
            _ = acquired.protected.cleanup(deadline=deadline)
        elif acquired.recorded:
            _end_unprotected(worker, acquired.context, deadline)
        if worker.process.poll() is None:
            _ = worker.process.wait(timeout=max(0, deadline - time.monotonic()))
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        _log.exception(
            "worker startup cleanup unresolved invocation=%s", worker.identity.invocation_id
        )
    finally:
        if acquired.protected is not None:
            with suppress(OSError):
                acquired.protected.close_lifeline()
        elif acquired.write_fd is not None:
            with suppress(OSError):
                os.close(acquired.write_fd)
        if acquired.helper is not None:
            with suppress(OSError, RuntimeError):
                stop_failed_helper(acquired.helper, deadline)
        _close_pipes(worker)
        if acquired.ticket is not None:
            acquired.ticket.close()


def _close_pipes(worker: WorkerProcess) -> None:
    for pipe in (worker.process.stdin, worker.process.stdout, worker.process.stderr):
        if pipe is not None:
            with suppress(OSError, ValueError):
                pipe.close()


def _end_unprotected(worker: WorkerProcess, context: ProtectionContext, deadline: float) -> None:
    evidence = context.evidence.with_deadline(deadline)
    record = evidence.get_worker(worker.identity.invocation_id)
    if record is None or record.ended_at is not None:
        return
    result = terminate_verified(worker.identity, record.descendants, deadline, worker.process)
    if result.covered_exited and record.observation_owner is None:
        evidence.end_worker(
            worker.identity.invocation_id, record.snapshot_seq, record.helper_generation
        )
