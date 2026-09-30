"""Own loop-worker startup, protection, streams, and cleanup."""

from __future__ import annotations

import logging
import os
import select
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import psutil

from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity
from milknado.loop._process_identity import Descendant, identity_state, observe_descendants
from milknado.loop._process_identity import terminate_verified as _terminate_verified
from milknado.loop._process_registry import LaunchTicket, WorkerRegistry

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


@dataclass(frozen=True, slots=True)
class SpawnOptions:
    command: tuple[str, ...]
    cwd: Path | None
    env: dict[str, str] | None
    text: bool
    stdin: int
    stdout: int | None
    stderr: int | None
    invocation_id: str | None = None


class WorkerProcess:
    def __init__(
        self,
        process: subprocess.Popen[str] | subprocess.Popen[bytes],
        identity: WorkerIdentity,
        gate_fd: int | None,
    ) -> None:
        self.process = process
        self.identity = identity
        self._gate_fd = gate_fd

    def release(self) -> None:
        if self._gate_fd is None:
            return
        gate_fd = self._gate_fd
        self._gate_fd = None
        try:
            os.write(gate_fd, b"R")
        finally:
            os.close(gate_fd)

    def close_gate(self) -> None:
        if self._gate_fd is not None:
            os.close(self._gate_fd)
            self._gate_fd = None


def spawn_gated(options: SpawnOptions) -> WorkerProcess:
    """Keep the actual command inert until its parent releases the exec gate."""
    if not options.command:
        raise ValueError("worker command is empty")
    gate_read: int | None = None
    gate_write: int | None = None
    command = options.command
    kwargs: dict[str, object] = {"start_new_session": os.name != "nt"}
    if os.name != "nt":
        gate_read, gate_write = os.pipe()
        command = (sys.executable, "-m", "milknado.loop._exec_gate", str(gate_read), *command)
        kwargs["pass_fds"] = (gate_read,)
    if options.text:
        kwargs.update(text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        proc = subprocess.Popen(  # pyright: ignore[reportCallIssue]
            command,
            stdin=options.stdin,
            stdout=options.stdout,
            stderr=options.stderr,
            cwd=options.cwd,
            env=options.env,
            **kwargs,  # pyright: ignore[reportArgumentType]
        )
    except Exception:
        if gate_write is not None:
            os.close(gate_write)
        raise
    finally:
        if gate_read is not None:
            os.close(gate_read)
    try:
        token = psutil.Process(proc.pid).create_time()
    except Exception:
        if gate_write is not None:
            os.close(gate_write)
        proc.kill()
        proc.wait(timeout=1)
        raise
    identity = WorkerIdentity(options.invocation_id or uuid.uuid4().hex, proc.pid, proc.pid, token)
    return WorkerProcess(proc, identity, gate_write)


class WorkerRecordView(Protocol):
    invocation_id: str
    snapshot_seq: int
    ready_generation: int
    observation_owner: str | None
    descendants: tuple[Descendant, ...]
    ended_at: str | None


class WorkerEvidence(Protocol):
    def record_worker(self, run_id: str, worker: WorkerIdentity) -> None: ...
    def get_worker(self, invocation_id: str) -> WorkerRecordView | None: ...
    def record_helper(self, helper: HelperIdentity) -> None: ...
    def begin_worker_observation(self, key: ObservationKey) -> None: ...
    def commit_worker_observation(
        self, key: ObservationKey, descendants: tuple[Descendant, ...]
    ) -> None: ...
    def end_worker(self, invocation_id: str, snapshot_seq: int) -> None: ...


@dataclass(frozen=True, slots=True)
class ProtectionContext:
    evidence: WorkerEvidence
    run_id: str
    db_path: Path
    registry: WorkerRegistry | None = None


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

    def _abort_protection(self) -> None:
        self._failed = True
        record = self._context.evidence.get_worker(self.identity.invocation_id)
        if record is None or record.ended_at is not None:
            return
        if record.observation_owner is not None:
            return
        try:
            if identity_state(self.identity.pid, self.identity.start_token) == "live":
                _snapshot(self.worker, self._context.evidence)
            record = self._context.evidence.get_worker(self.identity.invocation_id)
            if record is None or record.observation_owner is not None:
                return
            result = terminate_verified(
                self.identity, record.descendants, time.monotonic() + 3, self.process
            )
            if result.covered_exited:
                self._context.evidence.end_worker(self.identity.invocation_id, record.snapshot_seq)
        except (OSError, RuntimeError):
            _log.exception("worker protection cleanup unresolved invocation=%s", self.identity.invocation_id)

    def _replace_helper(self) -> bool:
        self._close_lifeline()
        deadline = time.monotonic() + 8
        while self._launches < 3 and time.monotonic() < deadline and not self._stop.is_set():
            try:
                record = self._context.evidence.get_worker(self.identity.invocation_id)
                if record is None or record.ended_at is not None:
                    return False
                sequence = record.snapshot_seq
                if identity_state(self.identity.pid, self.identity.start_token) == "live":
                    sequence = _snapshot(self.worker, self._context.evidence)
                self._launches += 1
                helper, write_fd = _start_helper(
                    self.worker, self._context, sequence, record.helper_generation + 1, deadline
                )
            except (OSError, RuntimeError) as exc:
                _log.warning("lifeline replacement failed invocation=%s: %s", self.identity.invocation_id, exc)
                if self._launches == 1:
                    self._abort_protection()
                    return False
                continue
            self._helper = helper
            self._write_fd = write_fd
            return True
        self._abort_protection()
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
                self._abort_protection()
                return

    def shutdown(self, deadline: float) -> bool:
        self._stop.set()
        if self._watch is not None:
            self._watch.join(timeout=max(0, deadline - time.monotonic()))
            if self._watch.is_alive():
                self._close_lifeline()
                return False
        try:
            record = self._context.evidence.get_worker(self.identity.invocation_id)
            if record is None or record.observation_owner is not None:
                return False
            if record.ended_at is None:
                if identity_state(self.identity.pid, self.identity.start_token) == "live":
                    _snapshot(self.worker, self._context.evidence)
                record = self._context.evidence.get_worker(self.identity.invocation_id)
                if record is None or record.observation_owner is not None:
                    return False
                result = terminate_verified(self.identity, record.descendants, deadline, self.process)
                if not result.covered_exited:
                    return False
                self._context.evidence.end_worker(self.identity.invocation_id, record.snapshot_seq)
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
        record = self._context.evidence.get_worker(self.identity.invocation_id)
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
        context.evidence.record_helper(identity)
        if helper.stdout is None:
            raise RuntimeError("lifeline READY stream missing")
        wait = 8 if deadline is None else max(0, min(8, deadline - time.monotonic()))
        ready, _, _ = select.select([helper.stdout], [], [], wait)
        expected = (
            f"READY {identity.invocation_id} {identity.generation} {identity.pid} "
            f"{identity.start_token} {sequence}"
        )
        if not ready or helper.stdout.readline().strip() != expected:
            raise RuntimeError("lifeline READY mismatch")
        record = context.evidence.get_worker(worker.identity.invocation_id)
        if record is None or record.ready_generation != identity.generation:
            raise RuntimeError("lifeline READY not durable")
        return helper, write_fd
    except Exception:
        os.close(write_fd)
        try:
            _ = helper.wait(timeout=3)
        except subprocess.TimeoutExpired:
            helper.kill()
            _ = helper.wait(timeout=1)
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
    try:
        context.evidence.record_worker(context.run_id, worker.identity)
        recorded = True
        sequence = _snapshot(worker, context.evidence)
        helper, write_fd = _start_helper(worker, context, sequence)
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
            record = context.evidence.get_worker(worker.identity.invocation_id)
            if record is not None and record.observation_owner is None and record.ended_at is None:
                result = terminate_verified(
                    worker.identity, record.descendants, time.monotonic() + 3, worker.process
                )
                if result.covered_exited:
                    context.evidence.end_worker(worker.identity.invocation_id, record.snapshot_seq)
        if worker.process.poll() is None:
            _ = worker.process.wait(timeout=3)
        if ticket is not None:
            ticket.close()
        raise
