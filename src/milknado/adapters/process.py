from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from milknado.domains.common import pid_alive
from milknado.domains.dispatch import ProcessOutcome


class ProcessAdapter:
    def __init__(self, *, termination_grace: float = 5.0, poll_interval: float = 0.1) -> None:
        self._termination_grace = termination_grace
        self._poll_interval = poll_interval

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
                    proc.communicate(input=stdin, timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
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
        return proc.pid

    def terminate_group(self, pid: int, timeout: float) -> bool:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(self._poll_interval)
        if pid_alive(pid):
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            deadline = time.monotonic() + timeout
            while pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(self._poll_interval)
        return not pid_alive(pid)

    @staticmethod
    def _write_stdin(stdin, payload: bytes) -> None:
        try:
            stdin.write(payload)
            stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def _terminate(self, proc: subprocess.Popen) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=self._termination_grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
