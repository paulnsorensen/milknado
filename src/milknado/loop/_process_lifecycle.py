"""Own loop-worker startup, protection, streams, and cleanup."""

from __future__ import annotations

import logging
import os
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import psutil

from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity
from milknado.loop._process_contract import ProtectionContext, WorkerEvidence
from milknado.loop._process_gate import SpawnOptions, WorkerProcess, spawn_gated
from milknado.loop._process_identity import Descendant, identity_state, observe_descendants
from milknado.loop._process_identity import terminate_verified as _terminate_verified
from milknado.loop._process_registry import LaunchTicket

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CleanupResult:
    covered_exited: bool
    unresolved: tuple[str, ...]


def terminate_verified(
    worker: WorkerIdentity,
    retained: tuple[Descendant, ...],
    deadline: float,
    process: subprocess.Popen[bytes] | subprocess.Popen[str] | None = None,
) -> CleanupResult:
    unresolved = _terminate_verified(worker, retained, deadline, process)
    return CleanupResult(not unresolved, unresolved)




class ProtectedWorker:
    def __init__(
        self, worker: WorkerProcess, helper: subprocess.Popen[str], write_fd: int,
        context: ProtectionContext,
    ) -> None:
        self.worker = worker
        self.process = worker.process
        self.identity = worker.identity
        self._helper = helper
        self._write_fd: int | None = write_fd
        self._context = context
        self._stop = threading.Event()
        self._watch: threading.Thread | None = None
        self._launches = 1
        self._failed = False
        self._ticket: LaunchTicket | None = None
        self._state_lock = threading.Lock()

    def _close_lifeline(self) -> None:
        with self._state_lock:
            write_fd, self._write_fd = self._write_fd, None
        if write_fd is not None:
            os.close(write_fd)

    def start_monitor(self) -> None:
        self._watch = threading.Thread(target=self._monitor, daemon=True)
        self._watch.start()

    def _abort_protection(self, deadline: float) -> None:
        self._failed = True
        self._close_lifeline()
        evidence = self._context.evidence.with_deadline(deadline)
        try:
            record = evidence.get_worker(self.identity.invocation_id)
            if record is None or record.ended_at is not None:
                return
            if record.observation_owner is None and identity_state(
                self.identity.pid, self.identity.start_token
            ) == "live":
                try:
                    _snapshot(self.worker, evidence)
                except (OSError, RuntimeError):
                    pass
                record = evidence.get_worker(self.identity.invocation_id)
                if record is None:
                    return
            result = terminate_verified(self.identity, record.descendants, deadline, self.process)
            if result.covered_exited and record.observation_owner is None:
                evidence.end_worker(
                    self.identity.invocation_id, record.snapshot_seq, record.helper_generation
                )
        except (OSError, RuntimeError):
            _log.exception("worker protection cleanup unresolved invocation=%s", self.identity.invocation_id)

    def _replace_helper(self) -> bool:
        self._close_lifeline()
        deadline = time.monotonic() + 8
        evidence = self._context.evidence.with_deadline(deadline)
        while self._launches < 3 and time.monotonic() < deadline and not self._stop.is_set():
            try:
                record = evidence.get_worker(self.identity.invocation_id)
                if record is None or record.ended_at is not None:
                    return False
                sequence = record.snapshot_seq
                if identity_state(self.identity.pid, self.identity.start_token) == "live":
                    sequence = _snapshot(self.worker, evidence)
                self._launches += 1
                helper, write_fd = _start_helper(
                    self.worker, self._context, sequence, record.helper_generation + 1, deadline
                )
            except (OSError, RuntimeError) as exc:
                _log.warning("lifeline replacement failed invocation=%s: %s", self.identity.invocation_id, exc)
                if self._launches == 1:
                    self._abort_protection(deadline)
                    return False
                continue
            self._helper = helper
            self._write_fd = write_fd
            return True
        self._abort_protection(deadline)
        return False

    def _monitor(self) -> None:
        while not self._stop.wait(0.2):
            if self._helper.poll() is not None:
                if not self._replace_helper():
                    return
                continue
            if self.process.poll() is not None:
                continue
            try:
                _snapshot(self.worker, self._context.evidence)
            except (OSError, RuntimeError):
                _log.exception("worker observation unresolved invocation=%s", self.identity.invocation_id)
                self._abort_protection(time.monotonic() + 3)
                return

    def shutdown(self, deadline: float) -> bool:
        self._stop.set()
        if self._watch is not None:
            self._watch.join(timeout=max(0, deadline - time.monotonic()))
            if self._watch.is_alive():
                self._close_lifeline()
                return False
        evidence = self._context.evidence.with_deadline(deadline)
        try:
            record = evidence.get_worker(self.identity.invocation_id)
            if record is None:
                return False
            if record.ended_at is None:
                if record.observation_owner is None and identity_state(
                    self.identity.pid, self.identity.start_token
                ) == "live":
                    _snapshot(self.worker, evidence)
                record = evidence.get_worker(self.identity.invocation_id)
                if record is None:
                    return False
                result = terminate_verified(self.identity, record.descendants, deadline, self.process)
                if not result.covered_exited or record.observation_owner is not None:
                    return False
                evidence.end_worker(
                    self.identity.invocation_id, record.snapshot_seq, record.helper_generation
                )
            if self._ticket is not None:
                self._ticket.close()
            return True
        except (OSError, RuntimeError):
            _log.exception("worker shutdown unresolved invocation=%s", self.identity.invocation_id)
            return False
        finally:
            self._close_lifeline()

    def finish(self, timeout: float = 3) -> bool:
        deadline = time.monotonic() + timeout
        self._stop.set()
        if self._watch is not None:
            self._watch.join(timeout=max(0, deadline - time.monotonic()))
            if self._watch.is_alive():
                return False
        self._close_lifeline()
        try:
            _ = self._helper.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return False
        record = self._context.evidence.with_deadline(deadline).get_worker(
            self.identity.invocation_id
        )
        confirmed = record is not None and record.ended_at is not None
        if confirmed and self._ticket is not None:
            self._ticket.close()
        return confirmed and not self._failed


def _snapshot(worker: WorkerProcess, evidence: WorkerEvidence) -> int:
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


def _start_helper(
    worker: WorkerProcess, context: ProtectionContext, sequence: int,
    generation: int = 0, deadline: float | None = None,
) -> tuple[subprocess.Popen[str], int]:
    limit = deadline if deadline is not None else time.monotonic() + 8
    read_fd, write_fd = os.pipe()
    try:
        helper = subprocess.Popen(
            (sys.executable, "-m", "milknado.adapters._loop_lifeline",
             str(context.db_path), str(read_fd), worker.identity.invocation_id, str(generation)),
            pass_fds=(read_fd,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except Exception:
        os.close(write_fd)
        raise
    finally:
        os.close(read_fd)
    try:
        identity = HelperIdentity(
            worker.identity.invocation_id, generation, helper.pid, psutil.Process(helper.pid).create_time()
        )
        evidence = context.evidence.with_deadline(limit)
        evidence.record_helper(identity)
        if helper.stdout is None:
            raise RuntimeError("lifeline READY stream missing")
        wait = max(0, limit - time.monotonic())
        ready, _, _ = select.select([helper.stdout], [], [], wait)
        expected = (
            f"READY {identity.invocation_id} {identity.generation} {identity.pid} "
            f"{identity.start_token} {sequence}"
        )
        if not ready or helper.stdout.readline().strip() != expected:
            raise RuntimeError("lifeline READY mismatch")
        record = evidence.get_worker(worker.identity.invocation_id)
        if (
            record is None or record.ended_at is not None
            or record.observation_owner is not None or record.snapshot_seq != sequence
            or record.ready_generation != identity.generation
            or record.helper_generation != identity.generation
            or record.helper_pid != identity.pid
            or record.helper_start_token != identity.start_token
        ):
            raise RuntimeError("lifeline READY not durable")
        return helper, write_fd
    except Exception:
        os.close(write_fd)
        try:
            _ = helper.wait(timeout=max(0, limit - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        raise


def spawn_protected(options: SpawnOptions, context: ProtectionContext) -> ProtectedWorker:
    ticket = context.registry.reserve() if context.registry is not None else None
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
    recorded = False
    deadline = time.monotonic() + 8
    evidence = context.evidence.with_deadline(deadline)
    try:
        evidence.record_worker(context.owner, worker.identity)
        recorded = True
        sequence = _snapshot(worker, evidence)
        helper, write_fd = _start_helper(worker, context, sequence, deadline=deadline)
        protected = ProtectedWorker(worker, helper, write_fd, context)
        protected._ticket = ticket
        if ticket is not None:
            if not ticket.activate(worker.release, protected.shutdown):
                raise RuntimeError("worker admission closed before READY")
        else:
            worker.release()
        protected.start_monitor()
        return protected
    except Exception:
        worker.close_gate()
        if recorded:
            record = evidence.get_worker(worker.identity.invocation_id)
            if record is not None and record.ended_at is None:
                result = terminate_verified(worker.identity, record.descendants, deadline, worker.process)
                if result.covered_exited and record.observation_owner is None:
                    evidence.end_worker(
                        worker.identity.invocation_id, record.snapshot_seq, record.helper_generation
                    )
        if worker.process.poll() is None:
            _ = worker.process.wait(timeout=3)
        if ticket is not None:
            ticket.close()
        raise
