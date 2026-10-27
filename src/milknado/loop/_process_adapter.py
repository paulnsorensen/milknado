from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from milknado.domains.common import WorkerIdentity, identity_state, pid_alive
from milknado.domains.dispatch import Descendant, ProcessOutcome, WorkerCleanupResult
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, StartEvidence
from milknado.loop._process_identity import observe_descendants
from milknado.loop._process_identity import terminate_verified_result as terminate_verified
from milknado.loop._process_lifecycle import spawn_protected
from milknado.loop._protection import protection_from_env


@dataclass(frozen=True, slots=True)
class _InlineLaunch:
    argv: tuple[str, ...]
    cwd: Path
    log_path: Path
    stdin: bytes
    env: dict[str, str]
    timeout: float
    run_id: str
    cancel_requested: Callable[[], bool] | None
    on_started: Callable[[int], None]


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
        if on_started is not None and (run_id := env.get("MILKNADO_RUN_ID")):
            return self._run_protected(
                _InlineLaunch(
                    argv, cwd, log_path, stdin, env, timeout, run_id, cancel_requested, on_started
                )
            )
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
                try:
                    on_started(proc.pid)
                except BaseException:
                    proc.kill()
                    _ = proc.communicate()
                    raise
            timed_out, cancelled = self._wait_for_exit(proc, stdin, timeout, cancel_requested)
            if timed_out or cancelled:
                if cancel_requested is None:
                    proc.kill()
                    _ = proc.wait()
                else:
                    self._terminate(proc)
        return ProcessOutcome(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    @staticmethod
    def _protection_context(launch: _InlineLaunch) -> ProtectionContext:
        return protection_from_env(launch.env, launch.run_id)

    def _run_protected(self, launch: _InlineLaunch) -> ProcessOutcome:
        context = self._protection_context(launch)
        with launch.log_path.open("wb") as log:
            options = SpawnOptions(
                launch.argv,
                launch.cwd,
                launch.env,
                False,
                subprocess.PIPE,
                log.fileno(),
                log.fileno(),
                launch.run_id,
            )
            worker = spawn_protected(options, context)
            try:
                if worker.worker.start_evidence is StartEvidence.EXECUTED:
                    launch.on_started(worker.process.pid)
                # SpawnOptions uses text=False, so this subprocess carries bytes.
                proc = cast(subprocess.Popen[bytes], worker.process)
                timed_out, cancelled = self._wait_for_exit(
                    proc, launch.stdin, launch.timeout, launch.cancel_requested
                )
                if not worker.complete(graceful=not (timed_out or cancelled)):
                    raise RuntimeError(f"worker cleanup unresolved for run {launch.run_id}")
            except BaseException as exc:
                if not worker.cleanup():
                    raise RuntimeError(
                        f"worker cleanup unresolved for run {launch.run_id}"
                    ) from exc
                raise
        if worker.worker.start_evidence is StartEvidence.STARTUP_FAILED:
            raise OSError(f"worker command failed before execution for run {launch.run_id}")
        if worker.worker.start_evidence is StartEvidence.UNKNOWN:
            raise RuntimeError(f"worker execution start unknown for run {launch.run_id}")
        return ProcessOutcome(
            worker.process.returncode if worker.process.returncode is not None else -1,
            timed_out,
            cancelled,
        )

    def _wait_for_exit(
        self,
        proc: subprocess.Popen[bytes],
        stdin: bytes,
        timeout: float,
        cancel_requested: Callable[[], bool] | None,
    ) -> tuple[bool, bool]:
        """Feed stdin and wait for exit; return (timed_out, cancelled). Callers terminate."""
        if cancel_requested is None:
            try:
                _ = proc.communicate(input=stdin, timeout=timeout)
            except subprocess.TimeoutExpired:
                return True, False
            return False, False
        if proc.stdin is not None:
            threading.Thread(
                target=self._write_stdin, args=(proc.stdin, stdin), daemon=True
            ).start()
        return self._wait(proc, timeout, cancel_requested)

    def _wait(
        self,
        proc: subprocess.Popen[bytes],
        timeout: float,
        cancel_requested: Callable[[], bool],
    ) -> tuple[bool, bool]:
        """Poll until exit, cancel, or deadline; return (timed_out, cancelled)."""
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if cancel_requested():
                return False, True
            if time.monotonic() >= deadline:
                return True, False
            time.sleep(self._poll_interval)
        return False, False

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
