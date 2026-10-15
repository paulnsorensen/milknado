"""Protected target execution and nonce-bound receipts for an inline tmux pane."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import NoReturn

import msgspec
import psutil

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner, default_config, load_config
from milknado.domains.dispatch import (
    ExecutedReceipt,
    FailedReceipt,
    FinishedReceipt,
    PreparedReceipt,
    Receipt,
)
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, StartEvidence
from milknado.loop._process_lifecycle import spawn_protected


def _receipt(path: Path, receipt: Receipt) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _ = stream.write(msgspec.json.encode(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        _ = temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _context() -> ProtectionContext:
    root = Path(os.environ["MILKNADO_PROJECT_ROOT"])
    config_path = root / "milknado.toml"
    config = load_config(config_path) if config_path.exists() else default_config(root)
    run_id = os.environ["MILKNADO_RUN_ID"]
    supervisor = psutil.Process()
    owner = WorkerOwner(
        run_id,
        supervisor.pid,
        supervisor.create_time(),
        run_id,
        int(os.environ["MILKNADO_NODE_ID"]),
    )
    return ProtectionContext(LoopWorkerEvidence(config.db_path), owner, config.db_path)


_PANE_SIGNALS = (signal.SIGTERM, signal.SIGHUP)
_SIGNAL_CHECK_SECS = 0.05


def _terminate(signum: int, _frame: object) -> NoReturn:
    for pane_signal in _PANE_SIGNALS:
        _ = signal.signal(pane_signal, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def _wait_for_exit(process: subprocess.Popen[str] | subprocess.Popen[bytes]) -> int:
    # Pane signals are process-directed. The kernel may deliver one to a lifecycle
    # monitor thread, and a main thread blocked in waitpid never runs its handler.
    while True:
        try:
            return process.wait(timeout=_SIGNAL_CHECK_SECS)
        except subprocess.TimeoutExpired:
            continue


def _run(nonce: str, path: Path, brief_name: str, command: tuple[str, ...]) -> int:
    for pane_signal in _PANE_SIGNALS:
        _ = signal.signal(pane_signal, _terminate)
    with Path(brief_name).open("rb") if brief_name != "-" else nullcontext(None) as brief:
        stdin = brief.fileno() if brief is not None else subprocess.DEVNULL
        options = SpawnOptions(command, None, None, False, stdin, None, subprocess.STDOUT)
        worker = spawn_protected(options, _context())
        try:
            evidence = worker.worker.start_evidence
            if evidence is StartEvidence.STARTUP_FAILED:
                _receipt(path, FailedReceipt(nonce, "target command failed before execution"))
            elif evidence is StartEvidence.EXECUTED:
                _receipt(
                    path, ExecutedReceipt(nonce, worker.process.pid, worker.identity.start_token)
                )
            if evidence is not StartEvidence.EXECUTED:
                if not worker.complete(graceful=evidence is StartEvidence.STARTUP_FAILED):
                    raise RuntimeError("tmux worker cleanup unresolved")
                return 125
            code = _wait_for_exit(worker.process)
            if not worker.complete(graceful=True):
                raise RuntimeError("tmux worker cleanup unresolved")
            _receipt(path, FinishedReceipt(nonce, worker.process.pid, code))
            return code if code >= 0 else 128 - code
        except BaseException as exc:
            if not worker.cleanup():
                raise RuntimeError("tmux worker cleanup unresolved") from exc
            raise


def main(arguments: list[str]) -> int:
    if len(arguments) < 4:
        return 125
    nonce, receipt_name, brief_name, *command = arguments
    path = Path(receipt_name)
    _receipt(path, PreparedReceipt(nonce))
    return _run(nonce, path, brief_name, tuple(command))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
