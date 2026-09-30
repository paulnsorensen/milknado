from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

from milknado.domains.common import WorkerIdentity, pid_alive
from milknado.domains.dispatch import Descendant, ProcessOutcome, WorkerCleanupResult
from milknado.loop._process_identity import identity_state, observe_descendants
from milknado.loop._process_identity import terminate_verified_result as terminate_verified


class ProcessAdapter:
    def __init__(self, *, termination_grace: float = 5.0, poll_interval: float = 0.1) -> None:
        self._termination_grace: float = termination_grace
        self._poll_interval: float = poll_interval

    def run(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        log_path: Path,
        stdin: bytes,
        env: dict[str, str],
        timeout: float,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        on_started: Callable[[int], None] | None = None,
    ) -> ProcessOutcome:
        timed_out = False
        cancelled = False
        with log_path.open("wb") as log:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
            )
            if on_started is not None:
                on_started(proc.pid)
            if cancel_requested is None:
                try:
                    _ = proc.communicate(input=stdin, timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    _ = proc.wait()
                    timed_out = True
            else:
                if proc.stdin is not None:
                    threading.Thread(
                        target=self._write_stdin,
                        args=(proc.stdin, stdin),
                        daemon=True,
                    ).start()
                deadline = time.monotonic() + timeout
                while proc.poll() is None:
                    if cancel_requested():
                        self._terminate(proc)
                        cancelled = True
                        break
                    if time.monotonic() >= deadline:
                        self._terminate(proc)
                        timed_out = True
                        break
                    time.sleep(self._poll_interval)
        return ProcessOutcome(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    def spawn_detached(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        log_path: Path,
        env: dict[str, str],
    ) -> int:
        with log_path.open("wb") as log:
            proc = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                start_new_session=True,
                env=env,
            )
        try:
            threading.Thread(
                target=proc.wait, name="milknado-detached-reaper", daemon=True
            ).start()
        except RuntimeError:
            proc.kill()
            _ = proc.wait()
            raise
        return proc.pid

    def terminate_group(self, pid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        grace_deadline = min(deadline - timeout / 2, time.monotonic() + self._termination_grace)
        while pid_alive(pid) and time.monotonic() < grace_deadline:
            time.sleep(max(0.0, min(self._poll_interval, grace_deadline - time.monotonic())))
        if pid_alive(pid):
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            while pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(max(0.0, min(self._poll_interval, deadline - time.monotonic())))
        return not pid_alive(pid)

    @staticmethod
    def supervisor_state(pid: int, start_token: float) -> str:
        return identity_state(pid, start_token)

    @staticmethod
    def observe_worker(worker: WorkerIdentity) -> tuple[Descendant, ...]:
        return observe_descendants(worker)

    @staticmethod
    def terminate_worker(
        worker: WorkerIdentity, retained: tuple[Descendant, ...], deadline: float
    ) -> WorkerCleanupResult:
        if time.monotonic() >= deadline:
            return WorkerCleanupResult(False, ("worker recovery deadline expired",))
        result = terminate_verified(worker, retained, deadline)
        return WorkerCleanupResult(result.covered_exited, result.unresolved)

    @staticmethod
    def _write_stdin(stdin: BinaryIO, payload: bytes) -> None:
        try:
            _ = stdin.write(payload)
            stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def _terminate(self, proc: subprocess.Popen[bytes]) -> None:
        proc.terminate()
        try:
            _ = proc.wait(timeout=self._termination_grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            _ = proc.wait()
