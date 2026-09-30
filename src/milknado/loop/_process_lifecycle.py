"""Own loop-worker startup, protection, streams, and cleanup."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from contextlib import suppress

from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, WorkerProcess
from milknado.loop._process_helper import HelperStart, UnconfirmedHelperExit
from milknado.loop._process_helper import start_helper as _start_helper
from milknado.loop._process_identity import identity_state
from milknado.loop._process_identity import terminate_verified_result as terminate_verified
from milknado.loop._process_observation import snapshot as _snapshot
from milknado.loop._process_registry import LaunchTicket
from milknado.loop._process_startup import spawn_protected as _spawn_protected

_log = logging.getLogger(__name__)


class ProtectedWorker:
    def __init__(
        self,
        worker: WorkerProcess,
        helper: subprocess.Popen[str],
        write_fd: int,
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
        self._monitor_started = threading.Event()
        self._replacements = 0
        self._failed = False
        self._ticket: LaunchTicket | None = None
        self._state_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._stop_deadline: float | None = None

    def _close_lifeline(self) -> None:
        with self._state_lock:
            write_fd, self._write_fd = self._write_fd, None
        if write_fd is not None:
            os.close(write_fd)

    def start_monitor(self) -> None:
        watch = threading.Thread(target=self._monitor, daemon=True)
        try:
            watch.start()
            self._watch = watch
        finally:
            self._monitor_started.set()

    def _abort_protection(self, deadline: float) -> None:
        self._failed = True
        self._close_lifeline()
        evidence = self._context.evidence.with_deadline(deadline)
        try:
            record = evidence.get_worker(self.identity.invocation_id)
            if record is None or record.ended_at is not None:
                return
            if (
                record.observation_owner is None
                and identity_state(self.identity.pid, self.identity.start_token) == "live"
            ):
                with suppress(OSError, RuntimeError):
                    _snapshot(self.worker, evidence)
                record = evidence.get_worker(self.identity.invocation_id)
                if record is None:
                    return
            result = terminate_verified(self.identity, record.descendants, deadline, self.process)
            if result.covered_exited and record.observation_owner is None:
                evidence.end_worker(
                    self.identity.invocation_id, record.snapshot_seq, record.helper_generation
                )
        except (OSError, RuntimeError):
            _log.exception(
                "worker protection cleanup unresolved invocation=%s", self.identity.invocation_id
            )

    def _abort_replacement(self) -> None:
        with self._state_lock:
            stop_deadline = self._stop_deadline
        deadline = time.monotonic() + 3
        if stop_deadline is not None:
            deadline = min(deadline, stop_deadline)
        self._abort_protection(deadline)

    def _replace_helper(self) -> bool:
        self._close_lifeline()
        deadline = time.monotonic() + 8
        evidence = self._context.evidence.with_deadline(deadline)
        while self._replacements < 3 and time.monotonic() < deadline and not self._stop.is_set():
            try:
                record = evidence.get_worker(self.identity.invocation_id)
                if record is None or record.ended_at is not None:
                    return False
                sequence = record.snapshot_seq
                if identity_state(self.identity.pid, self.identity.start_token) == "live":
                    sequence = _snapshot(self.worker, evidence)
            except (OSError, RuntimeError) as exc:
                _log.warning(
                    "lifeline evidence unavailable invocation=%s: %s",
                    self.identity.invocation_id,
                    exc,
                )
                self._abort_replacement()
                return False
            self._replacements += 1
            try:
                helper, write_fd = _start_helper(
                    self.worker,
                    self._context,
                    HelperStart(sequence, record.helper_generation + 1, deadline),
                )
            except UnconfirmedHelperExit:
                self._abort_replacement()
                return False
            except (OSError, RuntimeError) as exc:
                _log.warning(
                    "lifeline replacement failed invocation=%s: %s",
                    self.identity.invocation_id,
                    exc,
                )
                continue
            self._helper = helper
            self._write_fd = write_fd
            return True
        self._abort_replacement()
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
                _log.exception(
                    "worker observation unresolved invocation=%s", self.identity.invocation_id
                )
                self._abort_protection(time.monotonic() + 3)
                return

    def shutdown(self, deadline: float) -> bool:
        with self._state_lock:
            self._stop_deadline = (
                deadline if self._stop_deadline is None else min(deadline, self._stop_deadline)
            )
            deadline = self._stop_deadline
        self._stop.set()
        if not self._monitor_started.wait(timeout=max(0, deadline - time.monotonic())):
            self._close_lifeline()
            return False
        if not self._cleanup_lock.acquire(timeout=max(0, deadline - time.monotonic())):
            return False
        try:
            return self._shutdown_owned(deadline)
        finally:
            self._cleanup_lock.release()

    def _shutdown_owned(self, deadline: float) -> bool:
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
                if (
                    record.observation_owner is None
                    and identity_state(self.identity.pid, self.identity.start_token) == "live"
                ):
                    _snapshot(self.worker, evidence)
                record = evidence.get_worker(self.identity.invocation_id)
                if record is None:
                    return False
                result = terminate_verified(
                    self.identity, record.descendants, deadline, self.process
                )
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

    def complete(self, *, graceful: bool) -> bool:
        """Give a finished native session one bounded stdin-close grace period."""
        deadline = time.monotonic() + 3
        if graceful:
            if self.process.stdin is not None:
                with suppress(OSError, ValueError):
                    self.process.stdin.close()
            with suppress(subprocess.TimeoutExpired):
                _ = self.process.wait(timeout=min(0.5, max(0, deadline - time.monotonic())))
        return self.shutdown(deadline)

    def cleanup(
        self,
        threads: tuple[threading.Thread | None, ...] = (),
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> bool:
        """Stop the verified worker, drain readers, then close owned pipes."""
        if stop is not None:
            stop.set()
        limit = deadline if deadline is not None else time.monotonic() + 3
        with self._state_lock:
            if self._stop_deadline is not None:
                limit = min(limit, self._stop_deadline)
        confirmed = self.shutdown(limit)
        drained = True
        for thread in threads:
            if thread is not None:
                thread.join(timeout=max(0, limit - time.monotonic()))
                drained = drained and not thread.is_alive()
        if drained:
            for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                if pipe is not None:
                    with suppress(OSError, ValueError):
                        pipe.close()
        return confirmed and drained

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


def spawn_protected(options: SpawnOptions, context: ProtectionContext) -> ProtectedWorker:
    return _spawn_protected(options, context, _start_helper, ProtectedWorker)
