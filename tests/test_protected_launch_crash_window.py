"""A supervisor killed inside a protected launch keeps its retry claim fenced."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from milknado.adapters.process import ProcessAdapter
from milknado.domains.common import NodeStatus
from milknado.domains.dispatch.reconcile import fail_stale_running_runs
from milknado.domains.graph import MikadoGraph

if os.name != "posix":
    pytest.skip("POSIX process signals are required", allow_module_level=True)

_NOW = "2026-01-01T00:00:00+00:00"
_DEADLINE = 30
_SUPERVISOR = """\
import os
import signal
import subprocess
import sys
from pathlib import Path

import psutil

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.adapters._loop_worker_release import LoopWorkerRelease
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph, StartTaskAttempt
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions
from milknado.loop._process_lifecycle import spawn_protected

db, point, marker, now = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
graph = MikadoGraph(Path(db))
node = graph.add_node("task")
me = psutil.Process()
assert graph.claim_node(node.id, "owner", now=now, pid=me.pid)
_ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", None, 3))
graph.runs.start("owner", node.id, "run.log", now, 1, attempt_id="attempt")


def crash(invocation_id):
    marker.write_text(f"{node.id} {invocation_id}")
    os.kill(os.getpid(), signal.SIGKILL)


method, when = {
    "before-release": ("record_release", "before"),
    "after-release": ("record_release", "after"),
    "before-proof": ("confirm_execution", "before"),
}[point]
facade = type(graph.task_attempts)
real = getattr(facade, method)


def crashing(self, attempt_id, owner_run_id, invocation_id):
    if when == "before":
        crash(invocation_id)
    persisted = real(self, attempt_id, owner_run_id, invocation_id)
    assert persisted
    crash(invocation_id)


setattr(facade, method, crashing)
owner = WorkerOwner("owner", me.pid, me.create_time(), "owner", node.id)
context = ProtectionContext(
    LoopWorkerEvidence(Path(db)), owner, Path(db), release=LoopWorkerRelease(Path(db))
)
spawn_protected(
    SpawnOptions(
        (sys.executable, "-c", "pass"),
        Path(db).parent,
        None,
        True,
        subprocess.DEVNULL,
        subprocess.PIPE,
        subprocess.PIPE,
    ),
    context,
)
raise SystemExit("launch finished without reaching the crash window")
"""


def _kill_survivors(graph: MikadoGraph, invocation_id: str) -> None:
    record = graph.runs.get_worker(invocation_id)
    if record is None:
        return
    survivors = (
        (record.pid, record.start_token),
        (record.helper_pid, record.helper_start_token),
    )
    for pid, token in survivors:
        try:
            if pid is not None and psutil.Process(pid).create_time() == token:
                psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            continue


@pytest.mark.parametrize("point", ["before-release", "after-release", "before-proof"])
def test_supervisor_crash_in_protected_launch_keeps_claim_fenced(
    tmp_path: Path, point: str
) -> None:
    path, marker = tmp_path / "graph.db", tmp_path / "window.json"
    argv = [sys.executable, "-c", _SUPERVISOR, str(path), point, str(marker), _NOW]
    supervisor = subprocess.Popen(argv, stderr=subprocess.PIPE, text=True)
    graph: MikadoGraph | None = None
    invocation_id = ""
    try:
        _, stderr = supervisor.communicate(timeout=_DEADLINE)
        assert supervisor.returncode == -signal.SIGKILL, stderr
        node_text, invocation_id = marker.read_text().split()
        node_id = int(node_text)
        graph = MikadoGraph(path)
        assert fail_stale_running_runs(graph, node_id, ProcessAdapter()) == []
        assert not graph.try_reclaim(node_id, now=_NOW)
        assert not graph.claim_node(node_id, "next-owner", now=_NOW)
        current = graph.get_node(node_id)
        assert current is not None and current.status is NodeStatus.RUNNING
        assert current.run_id == "owner"
        assert graph.task_attempts.failed_count(node_id) == 0
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            _ = supervisor.communicate(timeout=5)
        if graph is not None:
            _kill_survivors(graph, invocation_id)
            graph.close()
