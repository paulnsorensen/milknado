"""Gate the direct child before it executes its worker command."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import final

import psutil

from milknado.domains.common import WorkerIdentity
from milknado.loop._output import SUBPROCESS_TEXT_KWARGS


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


@final
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
            _ = os.write(gate_fd, b"R")
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
            with suppress(OSError):
                os.close(gate_write)
        with suppress(OSError):
            proc.kill()
        with suppress(OSError, subprocess.TimeoutExpired):
            _ = proc.wait(timeout=1)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                with suppress(OSError, ValueError):
                    pipe.close()
        raise
    identity = WorkerIdentity(options.invocation_id or uuid.uuid4().hex, proc.pid, proc.pid, token)
    return WorkerProcess(proc, identity, gate_write)
