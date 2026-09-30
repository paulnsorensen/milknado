"""Real partial-start cleanup before a protected owner exists."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

import milknado.loop._process_lifecycle as lifecycle
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_gate import SpawnOptions, WorkerProcess
from milknado.loop._process_lifecycle import ProtectionContext, spawn_protected


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")
def test_helper_start_failure_closes_gated_worker_and_parent_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id)
    context = ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path)
    acquired: list[WorkerProcess] = []

    def fail_helper(worker: WorkerProcess, *_args: object) -> None:
        acquired.append(worker)
        raise RuntimeError("helper startup failed")

    monkeypatch.setattr(lifecycle, "_start_helper", fail_helper)
    try:
        with pytest.raises(RuntimeError, match="helper startup failed"):
            spawn_protected(
                SpawnOptions(
                    (sys.executable, "-c", "raise AssertionError('gate opened')"),
                    tmp_path,
                    None,
                    True,
                    subprocess.DEVNULL,
                    subprocess.PIPE,
                    subprocess.PIPE,
                ),
                context,
            )
        assert len(acquired) == 1
        worker = acquired[0]
        assert worker.process.poll() is not None
        assert worker.process.stdout is not None and worker.process.stdout.closed
        assert worker.process.stderr is not None and worker.process.stderr.closed
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.ended_at is not None
    finally:
        for worker in acquired:
            if worker.process.poll() is None:
                worker.process.kill()
                _ = worker.process.wait(timeout=1)
        graph.close()
