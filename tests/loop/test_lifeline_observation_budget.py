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

import milknado.loop._lifeline as lifeline
from milknado.domains.common import HelperIdentity, WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph, WorkerEvidenceStore
from milknado.loop._lifeline import _cleanup  # pyright: ignore[reportPrivateUsage]
from milknado.loop._process_identity import observe_descendants

pytestmark = pytest.mark.real_process


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_slow_observation_still_commits_and_closes_the_worker_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    supervisor = psutil.Process()
    helper = HelperIdentity("inv-1", 0, supervisor.pid, supervisor.create_time())
    try:
        graph.runs.record_worker(
            WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
            WorkerIdentity(
                "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
            ),
        )
        graph.runs.record_helper(helper)

        def slow_observe(identity: WorkerIdentity) -> tuple[tuple[int, float, int], ...]:
            time.sleep(lifeline._OBSERVATION_WAIT + 0.3)  # pyright: ignore[reportPrivateUsage]
            return observe_descendants(identity)

        monkeypatch.setattr(lifeline, "observe_descendants", slow_observe)
        reaper = threading.Thread(target=worker.wait, kwargs={"timeout": 30})
        reaper.start()
        with WorkerEvidenceStore(graph.db_path) as store:
            assert _cleanup(store, helper, time.monotonic() + 20) == 0
        reaper.join(timeout=5)
        assert worker.returncode is not None and worker.returncode != 0
        record = graph.runs.get_worker("inv-1")
        assert record is not None and record.ended_at is not None
        assert record.observation_owner is None
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            _ = worker.wait(timeout=2)
        graph.close()
