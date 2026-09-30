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
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, WorkerProcess, spawn_gated
from milknado.loop._process_lifecycle import spawn_protected
from milknado.loop._process_registry import WorkerRegistry


def _context(graph: MikadoGraph, node_id: int, registry: WorkerRegistry) -> ProtectionContext:
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node_id)
    return ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path, registry)


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
    assert selected.activate(lambda: None, lambda _: stopped.append("run-1") is None)
    assert unrelated.activate(lambda: None, lambda _: stopped.append("run-2") is None)

    assert registry.stop_run_workers("run-1", time.monotonic() + 1)
    assert stopped == ["run-1"]
    assert not unrelated.cancelled
