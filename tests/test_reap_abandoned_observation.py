"""A supervisor that dies between observation begin and commit must not strand its node."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import psutil
import pytest

from milknado.domains.common import ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.dispatch import ReapRequest, reap_orphaned_workers
from milknado.domains.graph import MikadoGraph, NodeWorkers, RunWorkers
from milknado.loop._process_adapter import ProcessAdapter

pytestmark = pytest.mark.real_process

if os.name != "posix":
    pytest.skip("POSIX process identities are required", allow_module_level=True)

# The real supervisor snapshot path: commit one observation, then stall inside the
# second one between begin and commit, where the issue #563 SIGKILL landed.
_SUPERVISOR = """
import os, subprocess, sys, time
from pathlib import Path
import psutil
from milknado.domains.common import WorkerIdentity, WorkerOwner
from milknado.loop import _process_observation
from milknado.loop._process_gate import WorkerProcess
from milknado.loop._worker_evidence import LoopWorkerEvidence

db, marker, node_id = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
)
identity = WorkerIdentity("inv-1", child.pid, child.pid, psutil.Process(child.pid).create_time())
supervisor = psutil.Process()
evidence = LoopWorkerEvidence(db)
owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node_id)
evidence.record_worker(owner, identity)
observations = []

def stall(worker):
    observations.append(worker)
    if len(observations) > 1:
        marker.with_suffix(".tmp").write_text(str(worker.pid))
        os.replace(marker.with_suffix(".tmp"), marker)
        time.sleep(60)
    return ()

_process_observation.observe_descendants = stall
worker = WorkerProcess(child, identity, None)
_process_observation.snapshot(worker, evidence)
_process_observation.snapshot(worker, evidence)
"""


@dataclass(frozen=True, slots=True)
class _Observing:
    graph: MikadoGraph
    node_id: int
    supervisor: subprocess.Popen[bytes]
    worker: psutil.Process


def _await_marker(marker: Path, supervisor: subprocess.Popen[bytes]) -> int:
    deadline = time.monotonic() + 20
    while not marker.exists():
        assert supervisor.poll() is None, "supervisor exited before observing"
        assert time.monotonic() < deadline, "supervisor never began its observation"
        time.sleep(0.02)
    return int(marker.read_text())


@pytest.fixture
def observing(tmp_path: Path) -> Iterator[_Observing]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("recover")
    assert graph.claim_node(node.id, "run-1", now="2026-01-01T00:00:00+00:00", pid=2**31 - 1)
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    marker = tmp_path / "worker.pid"
    command = [sys.executable, "-c", _SUPERVISOR, str(graph.db_path), str(marker), str(node.id)]
    supervisor = subprocess.Popen(command)
    worker: psutil.Process | None = None
    try:
        worker = psutil.Process(_await_marker(marker, supervisor))
        yield _Observing(graph, node.id, supervisor, worker)
    finally:
        supervisor.kill()
        _ = supervisor.wait(timeout=5)
        if worker is not None:
            with suppress(psutil.NoSuchProcess):
                worker.kill()
        graph.close()


def _stopped(worker: psutil.Process) -> bool:
    return not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE


def test_reap_takes_over_observation_abandoned_by_dead_supervisor(
    observing: _Observing,
) -> None:
    observing.supervisor.send_signal(signal.SIGKILL)
    _ = observing.supervisor.wait(timeout=5)
    stranded = observing.graph.runs.get_worker("inv-1")
    assert stranded is not None and stranded.ended_at is None
    assert (stranded.observation_owner, stranded.observation_seq) == ("supervisor", 2)
    assert stranded.snapshot_seq == 1

    request = ReapRequest(NodeWorkers(observing.node_id), deadline=time.monotonic() + 5)
    assert reap_orphaned_workers(observing.graph, ProcessAdapter(), request)

    assert _stopped(observing.worker)
    recorded = observing.graph.runs.get_worker("inv-1")
    assert recorded is not None and recorded.ended_at is not None
    assert recorded.observation_owner is None
    assert recorded.snapshot_seq == 2


def test_reap_keeps_observation_of_live_supervisor(
    observing: _Observing, caplog: pytest.LogCaptureFixture
) -> None:
    request = ReapRequest(RunWorkers("run-1"), deadline=time.monotonic() + 3)
    assert not reap_orphaned_workers(observing.graph, ProcessAdapter(), request)

    assert "interrupted observation=supervisor" in caplog.text
    assert observing.worker.is_running() and not _stopped(observing.worker)
    recorded = observing.graph.runs.get_worker("inv-1")
    assert recorded is not None and recorded.ended_at is None
    assert (recorded.observation_owner, recorded.observation_seq) == ("supervisor", 2)


def test_reap_keeps_observation_begun_by_another_observer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        node = graph.add_node("recover")
        assert graph.claim_node(node.id, "run-1", now="2026-01-01T00:00:00+00:00", pid=1)
        graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        start_token = psutil.Process(worker.pid).create_time()
        graph.runs.record_worker(
            WorkerOwner("run-1", 2**31 - 1, 123.5, "run-1", node.id),
            WorkerIdentity("inv-1", worker.pid, worker.pid, start_token),
        )
        reaper = ObservationKey("inv-1", "supervisor", 1, -1, worker.pid, start_token)
        graph.runs.begin_worker_observation(reaper)

        request = ReapRequest(NodeWorkers(node.id), deadline=time.monotonic() + 3)
        assert not reap_orphaned_workers(graph, ProcessAdapter(), request)

        assert "interrupted observation=supervisor" in caplog.text
        assert worker.poll() is None
        recorded = graph.runs.get_worker("inv-1")
        assert recorded is not None and recorded.ended_at is None
        assert (recorded.observation_owner, recorded.observation_seq) == ("supervisor", 1)
    finally:
        worker.kill()
        _ = worker.wait(timeout=5)
        graph.close()
