from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import HelperIdentity, WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph, WorkerRecord
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import WorkerProcess
from milknado.loop._process_identity import identity_state
from milknado.loop._process_lifecycle import ProtectedWorker
from milknado.loop._process_observation import snapshot


@pytest.fixture
def durable_worker(
    tmp_path: Path,
) -> Iterator[tuple[MikadoGraph, ProtectedWorker, int, float]]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    marker = tmp_path / "child-pid"
    program = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], "
        "start_new_session=True); Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", program, str(marker)], start_new_session=True, text=True
    )
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], text=True)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id)
    identity = WorkerIdentity(
        "inv-1", worker.pid, worker.pid, psutil.Process(worker.pid).create_time()
    )
    evidence = LoopWorkerEvidence(graph.db_path)
    protected = ProtectedWorker(
        WorkerProcess(worker, identity, None),
        helper,
        write_fd,
        ProtectionContext(evidence, owner, graph.db_path),
    )
    child_pid = 0
    try:
        graph.runs.record_worker(owner, identity)
        graph.runs.record_helper(
            HelperIdentity("inv-1", 0, helper.pid, psutil.Process(helper.pid).create_time())
        )
        limit = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < limit:
            time.sleep(0.02)
        child_pid = int(marker.read_text())
        _ = snapshot(protected.worker, evidence)
        durable = graph.runs.get_worker("inv-1")
        assert durable is not None and any(
            target[0] == child_pid for target in durable.descendants
        )
        yield graph, protected, child_pid, psutil.Process(child_pid).create_time()
    finally:
        protected.close_lifeline()
        if helper.poll() is None:
            helper.kill()
        _ = helper.wait(timeout=2)
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
        _ = worker.wait(timeout=2)
        if child_pid and psutil.pid_exists(child_pid):
            child = psutil.Process(child_pid)
            if child.status() != psutil.STATUS_ZOMBIE:
                child.kill()
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups required")
@pytest.mark.parametrize("cleanup", ["abort", "shutdown"])
def test_supervisor_failure_uses_last_durable_descendants(
    durable_worker: tuple[MikadoGraph, ProtectedWorker, int, float],
    monkeypatch: pytest.MonkeyPatch,
    cleanup: str,
) -> None:
    graph, protected, child_pid, child_token = durable_worker
    original_get = LoopWorkerEvidence.get_worker
    reads = 0

    def fail_later(store: LoopWorkerEvidence, invocation_id: str) -> WorkerRecord | None:
        nonlocal reads
        reads += 1
        if reads > 1:
            raise RuntimeError("later evidence unavailable")
        return original_get(store, invocation_id)

    monkeypatch.setattr(LoopWorkerEvidence, "get_worker", fail_later)
    reaper = threading.Thread(target=protected.process.wait, kwargs={"timeout": 4})
    reaper.start()
    if cleanup == "abort":
        protected._abort_protection(time.monotonic() + 3)  # pyright: ignore[reportPrivateUsage]
    else:
        assert not protected._shutdown_owned(time.monotonic() + 3)  # pyright: ignore[reportPrivateUsage]
    reaper.join(timeout=2)
    assert not reaper.is_alive()
    assert protected.process.returncode is not None and protected.process.returncode != 0
    assert identity_state(child_pid, child_token) == "gone"
    record = graph.runs.get_worker("inv-1")
    assert record is not None and record.ended_at is None
    assert any(target[0] == child_pid for target in record.descendants)
