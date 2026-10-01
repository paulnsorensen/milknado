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
from milknado.loop._process_identity import CleanupResult, Descendant, terminate_verified_result


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_helper_discovery_is_durable_for_later_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    marker = tmp_path / "detached-pid"
    program = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], "
        "start_new_session=True); Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )
    worker = subprocess.Popen([sys.executable, "-c", program, str(marker)], start_new_session=True)
    supervisor = psutil.Process()
    helper = HelperIdentity("inv-1", 0, supervisor.pid, supervisor.create_time())
    child_pid = 0
    try:
        graph.runs.record_worker(
            WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
            WorkerIdentity(
                "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
            ),
        )
        graph.runs.record_helper(helper)
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        child_pid = int(marker.read_text())

        def hold_termination(
            _worker: WorkerIdentity, _descendants: tuple[Descendant, ...], _deadline: float
        ) -> CleanupResult:
            return CleanupResult(False, ("held",))

        monkeypatch.setattr(lifeline, "terminate_verified", hold_termination)
        with WorkerEvidenceStore(graph.db_path) as store:
            assert _cleanup(store, helper, time.monotonic() + 3) == 1
        record = graph.runs.get_worker("inv-1")
        assert (
            record is not None
            and any(target[0] == child_pid for target in record.descendants)
            and record.observation_owner is None
            and record.ended_at is None
        )
        monkeypatch.setattr(lifeline, "terminate_verified", terminate_verified_result)
        reaper = threading.Thread(target=worker.wait, kwargs={"timeout": 3})
        reaper.start()
        with WorkerEvidenceStore(graph.db_path) as store:
            assert _cleanup(store, helper, time.monotonic() + 3) == 0
        reaper.join(timeout=2)
        assert not reaper.is_alive()
        assert not psutil.pid_exists(child_pid) or (
            psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            _ = worker.wait(timeout=2)
        if child_pid and psutil.pid_exists(child_pid):
            child = psutil.Process(child_pid)
            if child.status() != psutil.STATUS_ZOMBIE:
                child.kill()
        graph.close()
