"""Real protected-launch evidence never confuses gate release with exec."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph, StartTaskAttempt
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, StartEvidence, spawn_gated
from milknado.loop._process_startup import _release_worker  # pyright: ignore[reportPrivateUsage]


def _options(command: tuple[str, ...]) -> SpawnOptions:
    return SpawnOptions(
        command=command,
        cwd=None,
        env=None,
        text=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_confirmed_exec_keeps_worker_pid() -> None:
    worker = spawn_gated(_options((sys.executable, "-c", "import time; time.sleep(0.2)")))
    try:
        assert worker.process.pid == worker.identity.pid
        worker.release()
        assert worker.start_evidence is StartEvidence.EXECUTED
        assert worker.process.wait(timeout=3) == 0
    finally:
        worker.close_gate()


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_missing_command_after_release_is_startup_failure() -> None:
    worker = spawn_gated(_options(("/nonexistent/milknado-exec-test",)))
    try:
        worker.release()
        assert worker.start_evidence is StartEvidence.STARTUP_FAILED
        assert worker.process.wait(timeout=3) != 0
    finally:
        worker.close_gate()


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_signal_death_before_release_is_not_exec_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = spawn_gated(_options((sys.executable, "-c", "pass")))
    worker.process.kill()
    _ = worker.process.wait(timeout=3)
    gate_fd = worker._gate_fd  # pyright: ignore[reportPrivateUsage]
    status_fd = worker._status_fd  # pyright: ignore[reportPrivateUsage]
    assert gate_fd is not None and status_fd is not None

    def fail_write(_fd: int, _value: bytes) -> int:
        raise OSError("gate write failed")

    monkeypatch.setattr(os, "write", fail_write)
    worker.release()
    assert worker.start_evidence is StartEvidence.UNKNOWN
    for descriptor in (gate_fd, status_fd):
        with pytest.raises(OSError):
            _ = os.fstat(descriptor)


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_unlinked_release_skips_graph_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "absent.db"
    supervisor = psutil.Process()
    owner = WorkerOwner("runtime", supervisor.pid, supervisor.create_time())
    context = ProtectionContext(LoopWorkerEvidence(path), owner, path)

    def no_graph(_path: Path) -> MikadoGraph:
        pytest.fail("unlinked graph opened")

    monkeypatch.setattr("milknado.loop._process_startup.MikadoGraph", no_graph)
    worker = spawn_gated(_options((sys.executable, "-c", "pass")))
    try:
        _release_worker(worker, context)
        assert worker.start_evidence is StartEvidence.EXECUTED
        assert worker.process.wait(timeout=3) == 0
    finally:
        worker.close_gate()


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_corrupt_graph_prevents_worker_release(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.db"
    _ = path.write_bytes(b"not a SQLite database")
    supervisor = psutil.Process()
    owner = WorkerOwner("runtime", supervisor.pid, supervisor.create_time(), "linked")
    context = ProtectionContext(LoopWorkerEvidence(path), owner, path)
    marker = tmp_path / "executed"
    command = (sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()")
    worker = spawn_gated(_options(command))
    try:
        with pytest.raises(RuntimeError, match="graph run missing before worker release"):
            _release_worker(worker, context)
        worker.close_gate()
        assert worker.process.wait(timeout=3) != 0
        assert not marker.exists()
    finally:
        worker.close_gate()


def _linked_run(graph: MikadoGraph, owner: str, run_id: str, log: str) -> int:
    node = graph.add_node("task")
    assert graph.claim_node(node.id, owner, now="2026-01-01T00:00:00+00:00")
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", owner, run_id, 3))
    graph.runs.start(run_id, node.id, log, "2026-01-01T00:00:00+00:00", None, attempt_id="attempt")
    return node.id


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_protected_release_persists_bound_exec_proof(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node_id = _linked_run(graph, "owner", "worker-run", "worker.log")
    supervisor = psutil.Process()
    owner = WorkerOwner(
        "worker-run", supervisor.pid, supervisor.create_time(), "worker-run", node_id
    )
    context = ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path)
    worker = spawn_gated(_options((sys.executable, "-c", "pass")))
    try:
        _release_worker(worker, context)
        assert worker.process.wait(timeout=3) == 0
        record = graph.task_attempts.get("attempt")
        assert record is not None and record.start_state == "executed"
        assert record.active_invocation_id == worker.identity.invocation_id
    finally:
        worker.close_gate()
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_wrapper_error_after_readiness_is_not_startup_or_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from milknado.loop._exec_gate import main as exec_gate_main

    gate_read, gate_write = os.pipe()
    status_read, status_write = os.pipe()
    try:
        _ = os.write(gate_write, b"R")
        os.close(gate_write)
        monkeypatch.setattr(
            sys, "argv", ["exec-gate", str(gate_read), str(status_write), sys.executable]
        )

        def crash(*_args: object) -> None:
            raise ValueError("wrapper crashed before exec")

        monkeypatch.setattr(os, "execvpe", crash)
        assert exec_gate_main() == 72
        assert os.read(status_read, 3) == b"GPX"
    finally:
        os.close(status_read)


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
@pytest.mark.parametrize("protocol", [b"", b"PX", b"P"])
def test_untrusted_protocol_cannot_prove_execution(protocol: bytes) -> None:
    from milknado.domains.common import WorkerIdentity
    from milknado.loop._process_gate import WorkerProcess

    proc = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(72)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    token = psutil.Process(proc.pid).create_time()
    assert proc.wait(timeout=3) == 72
    status_read, status_write = os.pipe()
    try:
        if protocol:
            _ = os.write(status_write, protocol)
        os.close(status_write)
        worker = WorkerProcess(
            proc, WorkerIdentity("nonce", proc.pid, proc.pid, token), None, status_read
        )
        assert worker._observe_exec(status_read) is StartEvidence.UNKNOWN  # pyright: ignore[reportPrivateUsage]
    finally:
        os.close(status_read)


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_partial_readiness_without_eof_is_unknown() -> None:
    from milknado.domains.common import WorkerIdentity
    from milknado.loop._process_gate import WorkerProcess

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    token = psutil.Process(proc.pid).create_time()
    status_read, status_write = os.pipe()
    try:
        _ = os.write(status_write, b"P")
        worker = WorkerProcess(
            proc, WorkerIdentity("nonce", proc.pid, proc.pid, token), None, status_read
        )
        assert worker._observe_exec(status_read) is StartEvidence.UNKNOWN  # pyright: ignore[reportPrivateUsage]
    finally:
        os.close(status_write)
        os.close(status_read)
        proc.terminate()
        _ = proc.wait(timeout=3)


def _assert_current_review_process_can_prove_execution(path: Path, node_id: int) -> None:
    reopened = MikadoGraph(path)
    attempt = reopened.task_attempts.get("attempt")
    assert attempt is not None
    assert attempt.owner_run_id == "new-owner"
    assert attempt.active_invocation_id == "new-worker"
    assert attempt.start_state == "unknown"
    assert reopened.task_attempts.failed_count(node_id) == 0
    supervisor = psutil.Process()
    new_owner = WorkerOwner(
        "new-worker", supervisor.pid, supervisor.create_time(), "new-worker", node_id
    )
    current = spawn_gated(_options((sys.executable, "-c", "pass")))
    try:
        _release_worker(current, ProtectionContext(LoopWorkerEvidence(path), new_owner, path))
        assert current.process.wait(timeout=3) == 0
        confirmed = reopened.task_attempts.get("attempt")
        assert confirmed is not None
        assert confirmed.owner_run_id == "new-owner"
        assert confirmed.active_invocation_id == current.identity.invocation_id
        assert confirmed.start_state == "executed"
    finally:
        current.close_gate()
        reopened.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_delayed_review_worker_cannot_rebind_reopened_attempt(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node_id = _linked_run(graph, "old-owner", "old-worker", "old.log")
    supervisor = psutil.Process()
    owner = WorkerOwner(
        "old-worker", supervisor.pid, supervisor.create_time(), "old-worker", node_id
    )
    worker = spawn_gated(_options((sys.executable, "-c", "pass")))
    try:
        graph.runs.start(
            "new-worker",
            node_id,
            "new.log",
            "2026-01-01T00:00:00+00:00",
            None,
            attempt_id="attempt",
        )
        assert graph.task_attempts.rotate_review_round(
            "attempt", "old-owner", "new-owner", "new-worker"
        )
        graph.close()
        context = ProtectionContext(LoopWorkerEvidence(path), owner, path)
        with pytest.raises(RuntimeError, match="invocation fence lost"):
            _release_worker(worker, context)
        _assert_current_review_process_can_prove_execution(path, node_id)
    finally:
        worker.close_gate()
        if worker.process.poll() is None:
            worker.process.kill()
        _ = worker.process.wait(timeout=3)
