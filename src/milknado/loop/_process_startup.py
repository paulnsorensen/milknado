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

from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, WorkerProcess, spawn_gated
from milknado.loop._process_helper import HelperStart, _stop_failed_helper
from milknado.loop._process_identity import terminate_verified_result as terminate_verified
from milknado.loop._process_observation import snapshot as _snapshot
from milknado.loop._process_registry import LaunchTicket


class _ProtectedStart(Protocol):
    _ticket: LaunchTicket | None

    def start_monitor(self) -> None: ...
    def shutdown(self, deadline: float) -> bool: ...
    def cleanup(self, *, deadline: float) -> bool: ...
    def _close_lifeline(self) -> None: ...


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
        _ = worker.process.wait(timeout=3)
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
        protected._ticket = ticket
        if ticket is not None:
            if not ticket.activate(worker.release, protected.shutdown):
                raise RuntimeError("worker admission closed before READY")
        else:
            worker.release()
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
        if acquired.protected is not None:
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
                acquired.protected._close_lifeline()
        elif acquired.write_fd is not None:
            with suppress(OSError):
                os.close(acquired.write_fd)
        if acquired.helper is not None:
            with suppress(OSError, RuntimeError):
                _stop_failed_helper(acquired.helper, deadline)
        if acquired.ticket is not None:
            acquired.ticket.close()


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
