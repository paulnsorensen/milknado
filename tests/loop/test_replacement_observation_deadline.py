from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

import milknado.loop._process_lifecycle as lifecycle
import milknado.loop._process_observation as observation
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import HelperIdentity, WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import WorkerProcess
from milknado.loop._process_lifecycle import ProtectedWorker
from tests.loop.worker_release import PlainWorkerRelease


@pytest.fixture
def protected(tmp_path: Path) -> Iterator[tuple[MikadoGraph, ProtectedWorker]]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    worker = subprocess.Popen(sleeper, start_new_session=True, text=True)
    helper = subprocess.Popen(sleeper, text=True)
    _, write_fd = os.pipe()
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id)
    identity = WorkerIdentity(
        "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
    )
    graph.runs.record_worker(owner, identity)
    graph.runs.record_helper(
        HelperIdentity("inv-1", 0, helper.pid, psutil.Process(helper.pid).create_time())
    )
    context = ProtectionContext(
        LoopWorkerEvidence(graph.db_path), owner, graph.db_path, release=PlainWorkerRelease()
    )
    value = ProtectedWorker(WorkerProcess(worker, identity, None), helper, write_fd, context)
    try:
        yield graph, value
    finally:
        value.close_lifeline()
        for process in (helper, worker):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL) if process is worker else process.kill()
            _ = process.wait(timeout=2)
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_replacement_window_expiring_mid_observation_leaves_no_observation_owner(
    protected: tuple[MikadoGraph, ProtectedWorker], monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, worker = protected
    sample = observation.observe_descendants

    def slow_sample(identity: WorkerIdentity) -> tuple[object, ...]:
        time.sleep(0.6)
        return sample(identity)

    monkeypatch.setattr(lifecycle, "_REPLACEMENT_WINDOW", 0.3)
    monkeypatch.setattr(observation, "observe_descendants", slow_sample)
    worker._replace_helper()  # pyright: ignore[reportPrivateUsage]
    record = graph.runs.get_worker("inv-1")
    assert record is not None
    assert record.observation_owner is None
