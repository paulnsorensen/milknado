from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import milknado.loop._process_lifecycle as lifecycle
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import WorkerProcess
from milknado.loop._process_lifecycle import ProtectedWorker


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_periodic_observation_uses_one_second_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
        text=True,
    )
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], text=True)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), None, None)
    identity = WorkerIdentity(
        "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
    )
    protected = ProtectedWorker(
        WorkerProcess(worker, identity, None),
        helper,
        write_fd,
        ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path),
    )

    class PollClock:
        polls: int = 0

        def monotonic(self) -> float:
            return self.polls / 5

        def wait(self, timeout: float) -> bool:
            assert timeout == 0.2
            self.polls += 1
            return self.polls == 10

    clock = PollClock()
    snapshots: list[float] = []

    def record_snapshot(_worker: WorkerProcess, _evidence: LoopWorkerEvidence) -> int:
        snapshots.append(clock.monotonic())
        return 0

    try:
        monkeypatch.setattr(lifecycle, "time", SimpleNamespace(monotonic=clock.monotonic))
        monkeypatch.setattr(protected, "_stop", clock)
        monkeypatch.setattr(lifecycle, "_snapshot", record_snapshot)
        protected._monitor()  # pyright: ignore[reportPrivateUsage]
        assert clock.polls == 10
        assert snapshots == [1.0]
    finally:
        protected.close_lifeline()
        if helper.poll() is None:
            helper.kill()
        _ = helper.wait(timeout=2)
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
        _ = worker.wait(timeout=2)
        graph.close()
