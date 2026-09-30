from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

import milknado.loop._process_helper as process_helper
import milknado.loop._process_lifecycle as lifecycle
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_lifecycle import ProtectionContext, SpawnOptions, spawn_protected
from milknado.loop._process_registry import WorkerRegistry

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX helper uses passed file descriptors"
)


def _until(predicate, timeout: float = 9) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _context(
    graph: MikadoGraph, run_id: str, node_id: int, registry: WorkerRegistry | None = None
) -> ProtectionContext:
    supervisor = psutil.Process()
    owner = WorkerOwner(run_id, supervisor.pid, supervisor.create_time(), run_id, node_id)
    return ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path, registry)


def _graph(tmp_path: Path) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    return graph, node.id


def _options(tmp_path: Path) -> SpawnOptions:
    return SpawnOptions(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        tmp_path,
        None,
        False,
        subprocess.DEVNULL,
        subprocess.PIPE,
        subprocess.PIPE,
    )


def _ready(graph: MikadoGraph, invocation: str, generation: int) -> bool:
    record = graph.runs.get_worker(invocation)
    return record is not None and record.ready_generation == generation


def test_exhaustion_stops_only_affected_invocation(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path)
    other_node = graph.add_node("other")
    graph.runs.start("run-2", other_node.id, "other.log", "2026-01-01T00:00:00+00:00", None)
    affected = spawn_protected(_options(tmp_path), _context(graph, "run-1", node_id))
    other = spawn_protected(_options(tmp_path), _context(graph, "run-2", other_node.id))
    try:
        for generation in range(4):
            record = graph.runs.get_worker(affected.identity.invocation_id)
            assert record is not None and record.helper_pid is not None
            os.kill(record.helper_pid, signal.SIGKILL)
            if generation < 3:
                assert _until(
                    lambda expected=generation + 1: _ready(
                        graph, affected.identity.invocation_id, expected
                    )
                )
        assert _until(lambda: affected.process.poll() is not None)
        assert other.process.poll() is None
        assert _ready(graph, other.identity.invocation_id, 0)
        assert other.shutdown(time.monotonic() + 3)
    finally:
        for worker in (affected, other):
            if worker.process.poll() is None:
                worker.shutdown(time.monotonic() + 3)
        graph.close()


def test_mismatched_worker_identity_never_signals_live_process(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        token = psutil.Process(process.pid).create_time()
        wrong = WorkerIdentity("wrong", process.pid, process.pid, token + 1)
        result = lifecycle.terminate_verified(wrong, (), time.monotonic() + 1)
        assert not result.covered_exited
        assert any("mismatch" in issue for issue in result.unresolved)
        assert process.poll() is None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


def test_hung_replacement_keeps_finite_deadline_and_stops_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, node_id = _graph(tmp_path)
    worker = spawn_protected(_options(tmp_path), _context(graph, "run-1", node_id))
    fake = tmp_path / "hung-helper"
    fake.write_text("#!/bin/sh\nexec sleep 30\n")
    fake.chmod(0o700)
    helper_pid = 0
    try:
        before = graph.runs.get_worker(worker.identity.invocation_id)
        assert before is not None and before.helper_pid is not None
        monkeypatch.setattr(process_helper.sys, "executable", str(fake))
        start = time.monotonic()
        os.kill(before.helper_pid, signal.SIGKILL)
        assert _until(lambda: worker.process.poll() is not None, timeout=10)
        assert time.monotonic() - start < 9.5
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.ready_generation != 1
        assert record.helper_pid is not None
        helper_pid = record.helper_pid
        assert (
            not psutil.pid_exists(helper_pid)
            or psutil.Process(helper_pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        record = graph.runs.get_worker(worker.identity.invocation_id)
        helper_pid = helper_pid or (record.helper_pid if record is not None else 0)
        if helper_pid and psutil.pid_exists(helper_pid):
            helper = psutil.Process(helper_pid)
            if helper.status() != psutil.STATUS_ZOMBIE:
                helper.kill()
        if worker.process.poll() is None:
            worker.shutdown(time.monotonic() + 3)
        graph.close()


def test_shutdown_during_takeover_uses_first_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, node_id = _graph(tmp_path)
    registry = WorkerRegistry()
    context = _context(graph, "run-1", node_id, registry)
    worker = spawn_protected(_options(tmp_path), context)
    entered, release = threading.Event(), threading.Event()
    real_start = lifecycle._start_helper

    def held_start(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=4)
        return real_start(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "_start_helper", held_start)
    try:
        before = graph.runs.get_worker(worker.identity.invocation_id)
        assert before is not None and before.helper_pid is not None
        os.kill(before.helper_pid, signal.SIGKILL)
        assert entered.wait(timeout=3)
        deadline = time.monotonic() + 0.8
        start = time.monotonic()
        assert not registry.stop_all(deadline)
        assert time.monotonic() - start < 1.5
        assert not registry.stop_all(time.monotonic() + 4)
        assert time.monotonic() - start < 1.7
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.ended_at is None
    finally:
        release.set()
        if worker.process.poll() is None:
            worker.shutdown(time.monotonic() + 3)
        graph.close()


def test_worker_does_not_execute_before_initial_helper_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, node_id = _graph(tmp_path)
    marker = tmp_path / "worked"
    entered, release = threading.Event(), threading.Event()
    real_start = lifecycle._start_helper
    owned = []

    def held_start(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return real_start(*args, **kwargs)

    def launch() -> None:
        options = SpawnOptions(
            (sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
            tmp_path,
            None,
            False,
            subprocess.DEVNULL,
            subprocess.PIPE,
            subprocess.PIPE,
        )
        owned.append(spawn_protected(options, _context(graph, "run-1", node_id)))

    monkeypatch.setattr(lifecycle, "_start_helper", held_start)
    thread = threading.Thread(target=launch)
    try:
        thread.start()
        assert entered.wait(timeout=4)
        assert not marker.exists()
        records = graph.runs.live_workers(run_id="run-1")
        assert len(records) == 1 and records[0].ready_generation == -1
        release.set()
        thread.join(timeout=6)
        assert not thread.is_alive() and len(owned) == 1
        assert _until(marker.exists)
        assert owned[0].process.wait(timeout=3) == 0
        assert owned[0].finish(timeout=3)
    finally:
        release.set()
        thread.join(timeout=6)
        for worker in owned:
            if worker.process.poll() is None:
                worker.shutdown(time.monotonic() + 3)
        graph.close()


def test_repeated_pre_ready_helper_crashes_use_three_replacements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, node_id = _graph(tmp_path)
    worker = spawn_protected(_options(tmp_path), _context(graph, "run-1", node_id))
    fake = tmp_path / "crashing-helper"
    fake.write_text("#!/bin/sh\nexit 7\n")
    fake.chmod(0o700)
    try:
        before = graph.runs.get_worker(worker.identity.invocation_id)
        assert before is not None and before.helper_pid is not None
        monkeypatch.setattr(lifecycle.sys, "executable", str(fake))
        os.kill(before.helper_pid, signal.SIGKILL)
        assert _until(lambda: worker.process.poll() is not None)
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.helper_generation == 3
        assert record.ready_generation != 3
        assert record.ended_at is not None or record in graph.runs.live_workers(run_id="run-1")
    finally:
        if worker.process.poll() is None:
            worker.shutdown(time.monotonic() + 3)
        graph.close()
