from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import psutil
import pytest

import milknado.loop._process_lifecycle as lifecycle
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph, WorkerEvidenceStore, WorkerRecord
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, WorkerProcess
from milknado.loop._process_helper import HelperStart, UnconfirmedHelperExit
from milknado.loop._process_lifecycle import ProtectedWorker, spawn_protected


@pytest.fixture
def protected_record(tmp_path: Path) -> Iterator[tuple[MikadoGraph, ProtectedWorker]]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
        text=True,
    )
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], text=True)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id)
    identity = WorkerIdentity(
        "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
    )
    graph.runs.record_worker(owner, identity)
    graph.runs.record_helper(
        HelperIdentity("inv-1", 0, helper.pid, psutil.Process(helper.pid).create_time())
    )
    protected = ProtectedWorker(
        WorkerProcess(worker, identity, None),
        helper,
        write_fd,
        ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path),
    )
    try:
        yield graph, protected
    finally:
        protected.close_lifeline()
        if helper.poll() is None:
            helper.kill()
        _ = helper.wait(timeout=2)
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
        _ = worker.wait(timeout=2)
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_replacement_read_failure_stops_worker_and_preserves_record(
    protected_record: tuple[MikadoGraph, ProtectedWorker],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, protected = protected_record

    def fail_read(_evidence: LoopWorkerEvidence, _invocation_id: str) -> None:
        raise RuntimeError("evidence unavailable")

    monkeypatch.setattr(LoopWorkerEvidence, "get_worker", fail_read)
    assert not protected._replace_helper()  # pyright: ignore[reportPrivateUsage]
    assert protected._failed  # pyright: ignore[reportPrivateUsage]
    assert protected._replacements == 0  # pyright: ignore[reportPrivateUsage]
    assert protected._write_fd is None  # pyright: ignore[reportPrivateUsage]
    assert protected.process.poll() is not None
    record = graph.runs.get_worker("inv-1")
    assert record is not None and record.ended_at is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_shutdown_read_failure_stops_worker_and_preserves_record(
    protected_record: tuple[MikadoGraph, ProtectedWorker],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, protected = protected_record
    protected._monitor_started.set()  # pyright: ignore[reportPrivateUsage]

    def fail_read(_evidence: LoopWorkerEvidence, _invocation_id: str) -> None:
        raise RuntimeError("evidence unavailable")

    monkeypatch.setattr(LoopWorkerEvidence, "get_worker", fail_read)
    assert not protected.shutdown(time.monotonic() + 1)
    assert protected._write_fd is None  # pyright: ignore[reportPrivateUsage]
    assert protected.process.poll() is not None
    record = graph.runs.get_worker("inv-1")
    assert record is not None and record.ended_at is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_shutdown_keeps_uncertain_observation_open(
    protected_record: tuple[MikadoGraph, ProtectedWorker],
) -> None:
    graph, protected = protected_record
    protected._monitor_started.set()  # pyright: ignore[reportPrivateUsage]
    supervisor = psutil.Process()
    LoopWorkerEvidence(graph.db_path).begin_worker_observation(
        ObservationKey("inv-1", "supervisor", 1, 0, supervisor.pid, supervisor.create_time())
    )
    assert not protected.shutdown(time.monotonic() + 3)
    record = graph.runs.get_worker("inv-1")
    assert record is not None and record.ended_at is None
    assert record.observation_owner == "supervisor"
    assert protected.process.poll() is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_finish_timeout_does_not_close_open_record(
    protected_record: tuple[MikadoGraph, ProtectedWorker],
) -> None:
    graph, protected = protected_record
    assert not protected.finish(timeout=0.05)
    assert protected._write_fd is None  # pyright: ignore[reportPrivateUsage]
    assert protected.process.poll() is None
    record = graph.runs.get_worker("inv-1")
    assert record is not None and record.ended_at is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_replacement_abort_uses_existing_shutdown_deadline(
    protected_record: tuple[MikadoGraph, ProtectedWorker],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, protected = protected_record
    stop_deadline = time.monotonic() + 1
    protected._stop_deadline = stop_deadline  # pyright: ignore[reportPrivateUsage]
    deadlines: list[float] = []

    def fail_read(evidence: LoopWorkerEvidence, _invocation_id: str) -> None:
        assert evidence.deadline is not None
        deadlines.append(evidence.deadline)
        raise RuntimeError("evidence unavailable")

    monkeypatch.setattr(LoopWorkerEvidence, "get_worker", fail_read)
    protected._abort_replacement()  # pyright: ignore[reportPrivateUsage]
    assert deadlines == [stop_deadline]
    assert protected._failed  # pyright: ignore[reportPrivateUsage]
    assert protected.process.poll() is not None
    record = graph.runs.get_worker("inv-1")
    assert record is not None and record.ended_at is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_unconfirmed_replacement_aborts_owned_worker(
    protected_record: tuple[MikadoGraph, ProtectedWorker],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, protected = protected_record

    def fail_start(
        _worker: WorkerProcess, _context: ProtectionContext, _request: HelperStart
    ) -> NoReturn:
        raise UnconfirmedHelperExit("replacement exit unconfirmed")

    monkeypatch.setattr(lifecycle, "_start_helper", fail_start)
    assert not protected._replace_helper()  # pyright: ignore[reportPrivateUsage]
    assert protected._replacements == 1  # pyright: ignore[reportPrivateUsage]
    assert protected._failed  # pyright: ignore[reportPrivateUsage]
    assert protected.process.poll() is not None
    record = graph.runs.get_worker("inv-1")
    assert record is not None and record.ended_at is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_shutdown_missing_evidence_stops_known_worker_without_closing_record(
    protected_record: tuple[MikadoGraph, ProtectedWorker],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, protected = protected_record
    protected._monitor_started.set()  # pyright: ignore[reportPrivateUsage]

    def missing_read(_evidence: LoopWorkerEvidence, _invocation_id: str) -> None:
        return None

    monkeypatch.setattr(LoopWorkerEvidence, "get_worker", missing_read)
    assert not protected.shutdown(time.monotonic() + 1)
    assert protected.process.poll() is not None
    record = graph.runs.get_worker("inv-1")
    assert record is not None and record.ended_at is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_shutdown_waits_for_monitor_start_before_cleanup(
    protected_record: tuple[MikadoGraph, ProtectedWorker],
) -> None:
    graph, protected = protected_record
    assert not protected.shutdown(time.monotonic() + 0.02)
    assert protected._write_fd is None  # pyright: ignore[reportPrivateUsage]
    assert protected.process.poll() is None
    record = graph.runs.get_worker("inv-1")
    assert record is not None and record.ended_at is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_finish_waits_for_monitor_exit_before_claiming_completion(
    protected_record: tuple[MikadoGraph, ProtectedWorker],
) -> None:
    graph, protected = protected_record
    release = threading.Event()
    started = threading.Event()

    def delayed_monitor() -> None:
        started.set()
        assert release.wait(timeout=2)

    watch = threading.Thread(target=delayed_monitor)
    watch.start()
    assert started.wait(timeout=1)
    protected._watch = watch  # pyright: ignore[reportPrivateUsage]
    try:
        assert not protected.finish(timeout=0.02)
        assert protected.process.poll() is None
        record = graph.runs.get_worker("inv-1")
        assert record is not None and record.ended_at is None
    finally:
        release.set()
        watch.join(timeout=1)
        assert not watch.is_alive()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
@pytest.mark.parametrize("failure", ["missing", "error", "sqlite"])
def test_dead_helper_with_unavailable_evidence_stops_owned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id)
    protected = spawn_protected(
        SpawnOptions(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            tmp_path,
            None,
            True,
            subprocess.DEVNULL,
            subprocess.PIPE,
            subprocess.PIPE,
        ),
        ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path),
    )
    helper = protected._helper  # pyright: ignore[reportPrivateUsage]
    try:
        helper.kill()
        _ = helper.wait(timeout=2)

        def unavailable(_evidence: LoopWorkerEvidence, _invocation_id: str) -> WorkerRecord | None:
            if failure == "missing":
                return None
            raise RuntimeError("evidence unavailable")

        def locked(_store: WorkerEvidenceStore, _invocation_id: str) -> NoReturn:
            raise sqlite3.OperationalError("evidence locked")

        if failure == "sqlite":
            monkeypatch.setattr(WorkerEvidenceStore, "get", locked)
        else:
            monkeypatch.setattr(LoopWorkerEvidence, "get_worker", unavailable)
        watch = protected._watch  # pyright: ignore[reportPrivateUsage]
        assert watch is not None
        watch.join(timeout=5)
        assert not watch.is_alive()
        assert protected._failed  # pyright: ignore[reportPrivateUsage]
        assert protected.process.poll() is not None
        record = graph.runs.get_worker(protected.identity.invocation_id)
        assert record is not None and record.ended_at is None
    finally:
        if protected.process.poll() is None:
            os.killpg(protected.process.pid, signal.SIGKILL)
        _ = protected.process.wait(timeout=2)
        if helper.poll() is None:
            helper.kill()
        _ = helper.wait(timeout=2)
        graph.close()
