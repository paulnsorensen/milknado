"""Failure paths for the lifeline helper handshake."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import psutil
import pytest
from typing_extensions import override

import milknado.loop._process_helper as process_helper
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph, WorkerRecord
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, WorkerProcess, spawn_gated
from milknado.loop._process_helper import (
    HelperStart,
    _await_ready,  # pyright: ignore[reportPrivateUsage]
    stop_failed_helper,
)
from milknado.loop._process_observation import snapshot


@pytest.mark.skipif(os.name == "nt", reason="POSIX pipe semantics are required")
@pytest.mark.parametrize(
    "frame", [b"\xff\n", b"READY wrong\n", b"READY expected\nextra", b"x" * 256]
)
def test_ready_rejects_invalid_frame(frame: bytes) -> None:
    helper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,sys; os.write(1, bytes.fromhex(sys.argv[1]))",
            frame.hex(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert not _await_ready(helper, "READY expected", time.monotonic() + 2)
        assert helper.wait(timeout=2) == 0
    finally:
        if helper.poll() is None:
            helper.kill()
            _ = helper.wait(timeout=2)
        assert helper.stdout is not None
        helper.stdout.close()
        assert helper.stderr is not None
        helper.stderr.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX pipe semantics are required")
def test_ready_requires_stdout_pipe() -> None:
    helper = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert not _await_ready(helper, "READY expected", time.monotonic() + 2)
        assert helper.wait(timeout=2) == 0
    finally:
        if helper.poll() is None:
            helper.kill()
            _ = helper.wait(timeout=2)
        assert helper.stderr is not None
        helper.stderr.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals are required")
def test_failed_helper_escalates_when_sigterm_ignored(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    helper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,sys,time; from pathlib import Path; "
            + "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            + "Path(sys.argv[1]).touch(); time.sleep(30)",
            str(ready),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        stop_failed_helper(helper, deadline)
        assert helper.returncode == -signal.SIGKILL
        assert helper.stdout is not None and helper.stdout.closed
        assert helper.stderr is not None and helper.stderr.closed
    finally:
        if helper.poll() is None:
            helper.kill()
            _ = helper.wait(timeout=2)


@pytest.mark.skipif(os.name == "nt", reason="POSIX inherited pipe is required")
def test_helper_spawn_failure_closes_both_pipe_fds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pipe = os.pipe
    owned: list[int] = []

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        owned.extend(pair)
        return pair

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise OSError("helper spawn denied")

    monkeypatch.setattr(os, "pipe", tracked_pipe)
    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    with pytest.raises(OSError, match="helper spawn denied"):
        _ = process_helper.start_helper(
            cast(
                WorkerProcess,
                cast(object, SimpleNamespace(identity=SimpleNamespace(invocation_id="run"))),
            ),
            cast(ProtectionContext, cast(object, SimpleNamespace(db_path=Path("unused.db")))),
            HelperStart(0, 0, time.monotonic() + 2),
        )
    assert len(owned) == 2
    for fd in owned:
        with pytest.raises(OSError):
            _ = os.fstat(fd)


@dataclass(frozen=True, slots=True)
class _AdvanceAfterReady(LoopWorkerEvidence):
    worker: WorkerProcess = field(kw_only=True)

    @override
    def with_deadline(self, deadline: float) -> _AdvanceAfterReady:
        return _AdvanceAfterReady(self.db_path, deadline, worker=self.worker)

    @override
    def get_worker(self, invocation_id: str) -> WorkerRecord | None:
        _ = snapshot(self.worker, LoopWorkerEvidence(self.db_path, self.deadline))
        return LoopWorkerEvidence.get_worker(self, invocation_id)


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires inherited descriptors")
def test_ready_rejects_snapshot_advanced_after_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    current = psutil.Process()
    owner = WorkerOwner("run", current.pid, current.create_time(), "run", node.id)
    marker = tmp_path / "ran"
    command = (sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()")
    worker = spawn_gated(
        SpawnOptions(
            command, tmp_path, None, True, subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE
        )
    )
    evidence = LoopWorkerEvidence(graph.db_path)
    real_pipe = os.pipe
    owned: list[tuple[int, int]] = []

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        owned.append(pair)
        return pair

    try:
        evidence.record_worker(owner, worker.identity)
        sequence = snapshot(worker, evidence)
        context = ProtectionContext(
            _AdvanceAfterReady(graph.db_path, worker=worker), owner, graph.db_path
        )
        monkeypatch.setattr(os, "pipe", tracked_pipe)
        with pytest.raises(RuntimeError, match="lifeline READY not durable"):
            _ = process_helper.start_helper(
                worker, context, HelperStart(sequence, 0, time.monotonic() + 6)
            )
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.snapshot_seq == sequence + 1
        assert record.ready_generation == 0
        assert record.helper_pid is not None
        with pytest.raises(ChildProcessError):
            _ = os.waitpid(record.helper_pid, os.WNOHANG)
        assert owned
        for fd in owned[0]:
            with pytest.raises(OSError):
                _ = os.fstat(fd)
        assert not marker.exists()
    finally:
        worker.close_gate()
        if worker.process.poll() is None:
            _ = worker.process.wait(timeout=3)
        for pipe in (worker.process.stdout, worker.process.stderr):
            if pipe is not None:
                pipe.close()
        graph.close()
