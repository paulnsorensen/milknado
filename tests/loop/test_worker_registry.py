from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

import milknado.loop._process_startup as startup
import milknado.loop._worker_release as release_module
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, WorkerProcess, spawn_gated
from milknado.loop._process_lifecycle import spawn_protected
from milknado.loop._process_registry import WorkerRegistry
from milknado.loop._worker_evidence import LoopWorkerEvidence

pytestmark = pytest.mark.real_process


def _context(graph: MikadoGraph, node_id: int, registry: WorkerRegistry) -> ProtectionContext:
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node_id)
    return ProtectionContext(
        LoopWorkerEvidence(graph.db_path),
        owner,
        graph.db_path,
        registry,
        release=release_module.LoopWorkerRelease(graph.db_path),
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX protected launch is required")
def test_registry_stops_active_worker_without_dispatch_thread(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    registry = WorkerRegistry()
    worker = spawn_protected(
        SpawnOptions(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            tmp_path,
            None,
            False,
            subprocess.DEVNULL,
            subprocess.PIPE,
            subprocess.PIPE,
        ),
        _context(graph, node.id, registry),
    )
    try:
        assert registry.stop_all(time.monotonic() + 4)
        assert worker.process.poll() is not None
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        if worker.process.poll() is None:
            worker.process.kill()
            _ = worker.process.wait(timeout=1)
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")
def test_registry_abort_covers_popen_returning_after_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    registry = WorkerRegistry()
    entered, release = threading.Event(), threading.Event()
    real_spawn = spawn_gated

    def delayed_spawn(options: SpawnOptions) -> WorkerProcess:
        entered.set()
        assert release.wait(timeout=5)
        return real_spawn(options)

    monkeypatch.setattr(startup, "spawn_gated", delayed_spawn)
    marker = tmp_path / "worked"
    errors: list[Exception] = []

    def launch() -> None:
        try:
            _ = spawn_protected(
                SpawnOptions(
                    (
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).touch()",
                    ),
                    tmp_path,
                    None,
                    False,
                    subprocess.DEVNULL,
                    subprocess.PIPE,
                    subprocess.PIPE,
                ),
                _context(graph, node.id, registry),
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=launch)
    try:
        thread.start()
        assert entered.wait(timeout=5)
        assert registry.stop_all(time.monotonic() + 1) is False
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert not marker.exists()
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        release.set()
        thread.join(timeout=5)
        graph.close()


def _marker_options(path: Path, marker: Path) -> SpawnOptions:
    return SpawnOptions(
        (sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
        path,
        None,
        False,
        subprocess.DEVNULL,
        subprocess.PIPE,
        subprocess.PIPE,
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")
def test_registry_remains_available_during_graph_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    registry = WorkerRegistry()
    entered, resume = threading.Event(), threading.Event()
    marker = tmp_path / "worked"
    errors: list[Exception] = []

    def delayed_graph(path: Path) -> MikadoGraph:
        entered.set()
        assert resume.wait(timeout=5)
        return MikadoGraph(path)

    def launch() -> None:
        try:
            _ = spawn_protected(
                _marker_options(tmp_path, marker),
                _context(graph, node.id, registry),
            )
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(release_module, "MikadoGraph", delayed_graph)
    thread = threading.Thread(target=launch)
    try:
        thread.start()
        assert entered.wait(timeout=5)
        assert registry.lock.acquire(blocking=False)
        registry.lock.release()
        assert registry.stop_all(time.monotonic() + 1) is False
        resume.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(errors) == 1 and str(errors[0]) == "worker admission closed before READY"
        assert not marker.exists()
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        resume.set()
        thread.join(timeout=5)
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")
def test_stop_during_gate_release_confirms_the_released_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    registry = WorkerRegistry()
    stopped: list[bool] = []
    stopper = threading.Thread(
        target=lambda: stopped.append(registry.stop_all(time.monotonic() + 8))
    )
    real_release = WorkerProcess.release
    released: list[WorkerProcess] = []

    def release_then_stop(worker: WorkerProcess) -> None:
        # A signal that lands after exec but before activation finishes.
        real_release(worker)
        released.append(worker)
        stopper.start()
        deadline = time.monotonic() + 5
        while not registry.requested() and time.monotonic() < deadline:
            time.sleep(0.01)

    monkeypatch.setattr(WorkerProcess, "release", release_then_stop)
    try:
        with pytest.raises(RuntimeError, match="admission closed before READY"):
            _ = spawn_protected(_sleeper_options(tmp_path), _context(graph, node.id, registry))
        stopper.join(timeout=10)
        assert stopped == [True]
        assert released[0].process.poll() is not None
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        for worker in released:
            if worker.process.poll() is None:
                worker.process.kill()
                _ = worker.process.wait(timeout=1)
        stopper.join(timeout=10)
        graph.close()


def _sleeper_options(path: Path) -> SpawnOptions:
    return SpawnOptions(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        path,
        None,
        False,
        subprocess.DEVNULL,
        subprocess.PIPE,
        subprocess.PIPE,
    )


def test_registry_rejects_admission_on_scalar_intent() -> None:
    registry = WorkerRegistry()
    requested = False
    registry.bind_shutdown_intent(lambda: requested)
    ticket = registry.reserve()
    requested = True
    assert not ticket.activate(lambda: pytest.fail("released after signal"), lambda _: True)
    with pytest.raises(RuntimeError, match="admission is closed"):
        _ = registry.reserve()
    ticket.close()


def test_registry_attempts_all_active_workers_with_one_deadline() -> None:
    registry = WorkerRegistry()
    first = registry.reserve()
    second = registry.reserve()
    barrier = threading.Barrier(2)

    def shutdown(_deadline: float) -> bool:
        _ = barrier.wait(timeout=0.5)
        return True

    assert first.activate(lambda: None, shutdown)
    assert second.activate(lambda: None, shutdown)
    assert registry.stop_all(time.monotonic() + 0.5)


def test_registry_stops_only_tickets_for_graph_run() -> None:
    registry = WorkerRegistry()
    selected = registry.reserve("run-1")
    unrelated = registry.reserve("run-2")
    stopped: list[str] = []
    selected_active = selected.activate(lambda: None, lambda _: stopped.append("run-1") is None)
    unrelated_active = unrelated.activate(lambda: None, lambda _: stopped.append("run-2") is None)
    assert selected_active
    assert unrelated_active

    assert registry.stop_run_workers("run-1", time.monotonic() + 1)
    assert stopped == ["run-1"]
    assert not unrelated.cancelled
