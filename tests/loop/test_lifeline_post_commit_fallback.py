from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph, WorkerEvidenceStore, WorkerRecord
from milknado.loop._lifeline import _cleanup  # pyright: ignore[reportPrivateUsage]
from milknado.loop._process_identity import identity_state


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
def test_post_commit_read_failure_cleans_last_durable_targets(
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
    reaper = threading.Thread(target=worker.wait)
    try:
        graph.runs.record_worker(
            WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id),
            WorkerIdentity("inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()),
        )
        graph.runs.record_helper(helper)
        limit = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < limit:
            time.sleep(0.02)
        child_pid = int(marker.read_text())
        child_token = psutil.Process(child_pid).create_time()
        key = ObservationKey("inv-1", "supervisor", 1, 0, supervisor.pid, supervisor.create_time())
        graph.runs.begin_worker_observation(key)
        graph.runs.commit_worker_observation(key, ((child_pid, child_token, child_pid),))
        original_get = WorkerEvidenceStore.get
        reads = 0

        def fail_second(store: WorkerEvidenceStore, invocation_id: str) -> WorkerRecord | None:
            nonlocal reads
            reads += 1
            if reads == 2:
                raise sqlite3.OperationalError("post-commit read unavailable")
            return original_get(store, invocation_id)

        reaper.start()
        with monkeypatch.context() as patch:
            patch.setattr(WorkerEvidenceStore, "get", fail_second)
            with WorkerEvidenceStore(graph.db_path) as store:
                assert _cleanup(store, helper, time.monotonic() + 3) == 1
        assert reads == 2
        reaper.join(timeout=2)
        assert not reaper.is_alive()
        assert worker.returncode is not None and worker.returncode != 0
        assert identity_state(child_pid, child_token) == "gone"
        record = graph.runs.get_worker("inv-1")
        assert record is not None and record.ended_at is None
        assert record.snapshot_seq == 2
        assert any(target[0] == child_pid for target in record.descendants)
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
        if reaper.is_alive():
            reaper.join(timeout=2)
        elif worker.returncode is None:
            _ = worker.wait(timeout=2)
        if child_pid and psutil.pid_exists(child_pid):
            child = psutil.Process(child_pid)
            if child.status() != psutil.STATUS_ZOMBIE:
                child.kill()
        graph.close()
