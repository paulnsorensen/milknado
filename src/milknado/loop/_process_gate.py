"""Gate the direct child before it executes its worker command."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import final

import psutil

from milknado.domains.common import WorkerIdentity
from milknado.loop._output import SUBPROCESS_TEXT_KWARGS
from milknado.loop._process_identity import identity_state

# Bounds interpreter startup of the gate (until it reports "G"), not worker exec.
# Generous because a loaded host can take seconds to start Python.
_GATE_STARTUP_TIMEOUT = 30.0
# Bounds read, report, and exec after release. One second proved too short on a
# loaded host: a late report left a launched worker as unknown start evidence.
_EXEC_REPORT_TIMEOUT = 10.0


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


class StartEvidence(StrEnum):
    UNKNOWN = "unknown"
    EXECUTED = "executed"
    STARTUP_FAILED = "startup_failed"


@final
class WorkerProcess:
    def __init__(
        self,
        process: subprocess.Popen[str] | subprocess.Popen[bytes],
        identity: WorkerIdentity,
        gate_fd: int | None,
        status_fd: int | None = None,
    ) -> None:
        self.process = process
        self.identity = identity
        self._gate_fd = gate_fd
        self._status_fd = status_fd
        self.start_evidence = (
            StartEvidence.EXECUTED if status_fd is None else StartEvidence.UNKNOWN
        )

    def _exited_after_exec(self) -> StartEvidence:
        code = self.process.poll()
        return (
            StartEvidence.EXECUTED
            if code is not None and code >= 0 and code != 72
            else StartEvidence.UNKNOWN
        )

    def _confirm_exec(self) -> StartEvidence:
        if self.process.poll() is not None:
            return self._exited_after_exec()
        if identity_state(self.identity.pid, self.identity.start_token) == "live":
            return StartEvidence.EXECUTED
        # A short worker can exit between the poll and the identity check.
        return self._exited_after_exec()

    def _observe_exec(self, status_fd: int) -> StartEvidence:
        deadline = time.monotonic() + _EXEC_REPORT_TIMEOUT
        protocol = b""
        while time.monotonic() < deadline:
            try:
                ready, _, _ = select.select(
                    [status_fd], [], [], max(0, deadline - time.monotonic())
                )
            except (OSError, ValueError):
                break
            if not ready:
                break
            try:
                chunk = os.read(status_fd, 3 - len(protocol))
            except OSError:
                break
            if not chunk:
                if protocol != b"P":
                    break
                return self._confirm_exec()
            protocol += chunk
            if protocol == b"PF":
                return StartEvidence.STARTUP_FAILED
            if protocol != b"P":
                break
        return StartEvidence.UNKNOWN

    @staticmethod
    def _await_gate_ready(status_fd: int) -> bool:
        # The gate reports "G" once its interpreter is up. Nothing executes before
        # release, so a gate that never reports G has not run the worker.
        try:
            ready, _, _ = select.select([status_fd], [], [], _GATE_STARTUP_TIMEOUT)
            reported = os.read(status_fd, 1) if ready else None
        except (OSError, ValueError):
            return False
        if reported is None:
            raise TimeoutError
        return reported == b"G"

    def _fail_startup(self) -> None:
        # R was never written, so the worker command never executed.
        with suppress(OSError):
            self.process.kill()
        with suppress(OSError, subprocess.TimeoutExpired):
            _ = self.process.wait(timeout=1)
        self.start_evidence = StartEvidence.STARTUP_FAILED

    def release(self) -> None:
        if self._gate_fd is None:
            return
        gate_fd = self._gate_fd
        self._gate_fd = None
        status_fd = self._status_fd
        self._status_fd = None
        try:
            ready = status_fd is None or self._await_gate_ready(status_fd)
        except TimeoutError:
            self._fail_startup()
            ready = False
        try:
            if ready:
                with suppress(OSError):
                    _ = os.write(gate_fd, b"R")
        finally:
            os.close(gate_fd)
        if status_fd is not None:
            try:
                if ready:
                    self.start_evidence = self._observe_exec(status_fd)
            finally:
                os.close(status_fd)

    def close_gate(self) -> None:
        if self._gate_fd is not None:
            os.close(self._gate_fd)
            self._gate_fd = None
        if self._status_fd is not None:
            os.close(self._status_fd)
            self._status_fd = None


def _close_fds(*descriptors: int | None) -> None:
    for descriptor in descriptors:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _abort_unidentified_worker(
    proc: subprocess.Popen[str] | subprocess.Popen[bytes],
    gate_write: int | None,
    status_read: int | None,
) -> None:
    _close_fds(gate_write, status_read)
    with suppress(OSError):
        proc.kill()
    with suppress(OSError, subprocess.TimeoutExpired):
        _ = proc.wait(timeout=1)
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            with suppress(OSError, ValueError):
                pipe.close()


def spawn_gated(options: SpawnOptions) -> WorkerProcess:
    """Keep the actual command inert until its parent releases the exec gate."""
    if not options.command:
        raise ValueError("worker command is empty")
    gate_read: int | None = None
    gate_write: int | None = None
    status_read: int | None = None
    status_write: int | None = None
    command = options.command
    kwargs: dict[str, object] = {"start_new_session": os.name != "nt"}
    if os.name != "nt":
        gate_read, gate_write = os.pipe()
        status_read, status_write = os.pipe()
        command = (
            sys.executable,
            "-m",
            "milknado.loop._exec_gate",
            str(gate_read),
            str(status_write),
            *command,
        )
        kwargs["pass_fds"] = (gate_read, status_write)
    if options.text:
        kwargs.update(SUBPROCESS_TEXT_KWARGS)
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
        _close_fds(gate_write, status_read)
        raise
    finally:
        if gate_read is not None:
            os.close(gate_read)
        if status_write is not None:
            os.close(status_write)
    try:
        token = psutil.Process(proc.pid).create_time()
    except Exception:
        _abort_unidentified_worker(proc, gate_write, status_read)
        raise
    identity = WorkerIdentity(options.invocation_id or uuid.uuid4().hex, proc.pid, proc.pid, token)
    return WorkerProcess(proc, identity, gate_write, status_read)
