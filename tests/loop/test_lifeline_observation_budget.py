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
from milknado.loop._process_identity import observe_descendants

pytestmark = pytest.mark.real_process

_SLOW_OBSERVE = lifeline._OBSERVATION_WAIT + 0.3  # pyright: ignore[reportPrivateUsage]


def _slow_observe(identity: WorkerIdentity) -> tuple[tuple[int, float, int], ...]:
    time.sleep(_SLOW_OBSERVE)
    return observe_descendants(identity)


def _record_worker(graph: MikadoGraph, worker: subprocess.Popen[bytes]) -> HelperIdentity:
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    supervisor = psutil.Process()
    helper = HelperIdentity("inv-1", 0, supervisor.pid, supervisor.create_time())
    graph.runs.record_worker(
        WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
        WorkerIdentity("inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()),
    )
    graph.runs.record_helper(helper)
    return helper


def _cleanup(graph: MikadoGraph, helper: HelperIdentity, budget: float) -> int:
    with WorkerEvidenceStore(graph.db_path) as store:
        return lifeline._cleanup(store, helper, time.monotonic() + budget)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
@pytest.mark.parametrize("budget", [20.0, _SLOW_OBSERVE - 0.2], ids=["ample", "spent"])
def test_slow_observation_still_commits_and_releases_the_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: float
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        helper = _record_worker(graph, worker)
        monkeypatch.setattr(lifeline, "observe_descendants", _slow_observe)
        reaper = threading.Thread(target=worker.wait, kwargs={"timeout": 30})
        reaper.start()
        result = _cleanup(graph, helper, budget)
        record = graph.runs.get_worker("inv-1")
        assert record is not None and record.observation_owner is None
        if budget > _SLOW_OBSERVE:
            assert result == 0
            reaper.join(timeout=5)
            assert worker.returncode is not None and worker.returncode != 0
            assert record.ended_at is not None
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            _ = worker.wait(timeout=2)
        graph.close()
