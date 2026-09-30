from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import msgspec
import psutil
import pytest

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions
from milknado.loop._process_lifecycle import ProtectedWorker, spawn_protected

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX helper uses passed file descriptors"
)


class _SupervisorFacts(msgspec.Struct, frozen=True):
    invocation: str
    pid: int
    helper: int


_SUPERVISOR = """
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.loop._process_lifecycle import ProtectionContext, SpawnOptions, spawn_protected

db_path, marker = map(Path, sys.argv[1:3])
supervisor = psutil.Process()
owner = WorkerOwner('run-1', supervisor.pid, supervisor.create_time(), 'run-1', int(sys.argv[3]))
context = ProtectionContext(LoopWorkerEvidence(db_path), owner, db_path)
worker = spawn_protected(SpawnOptions(
    (sys.executable, '-c', 'import time; time.sleep(60)'),
    db_path.parent, None, False, subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE,
), context)
record = context.evidence.get_worker(worker.identity.invocation_id)
marker.write_text(json.dumps({'invocation': worker.identity.invocation_id,
                              'pid': worker.process.pid, 'helper': record.helper_pid}))
while True:
    time.sleep(1)
"""


def _until(predicate: Callable[[], bool], timeout: float = 8) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _alive(pid: int, token: float) -> bool:
    try:
        process = psutil.Process(pid)
        return (
            process.create_time() == token
            and process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def _graph(tmp_path: Path) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    return graph, node.id


def _protected(
    graph: MikadoGraph,
    node_id: int,
    program: str,
    interactive: bool = False,
) -> ProtectedWorker:
    supervisor = psutil.Process()
    context = ProtectionContext(
        LoopWorkerEvidence(graph.db_path),
        WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node_id),
        graph.db_path,
    )
    options = SpawnOptions(
        (sys.executable, "-c", program),
        graph.db_path.parent,
        None,
        interactive,
        subprocess.PIPE if interactive else subprocess.DEVNULL,
        subprocess.PIPE,
        subprocess.PIPE,
    )
    return spawn_protected(options, context)


@pytest.mark.parametrize("replace", [False, True], ids=["initial-ready", "replacement-ready"])
def test_ready_lifeline_reaps_worker_after_supervisor_sigkill(
    tmp_path: Path,
    replace: bool,
) -> None:
    graph, node_id = _graph(tmp_path)
    marker = tmp_path / "ready.json"
    supervisor = subprocess.Popen(
        (sys.executable, "-c", _SUPERVISOR, str(graph.db_path), str(marker), str(node_id)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid = 0
    token = 0.0
    try:
        assert _until(marker.exists), "supervisor did not publish protected worker readiness"
        facts = msgspec.json.decode(marker.read_bytes(), type=_SupervisorFacts)
        record = graph.runs.get_worker(facts.invocation)
        assert record is not None and record.ready_generation == 0
        pid, token = record.pid, record.start_token
        assert _alive(pid, token)
        if replace:
            os.kill(facts.helper, signal.SIGKILL)
            assert _until(
                lambda: (
                    (current := graph.runs.get_worker(facts.invocation)) is not None
                    and current.ready_generation == 1
                    and current.helper_pid != facts.helper
                )
            ), "replacement never became durably ready"
            assert _alive(pid, token)
        os.kill(supervisor.pid, signal.SIGKILL)
        _ = supervisor.wait(timeout=2)
        assert _until(lambda: not _alive(pid, token), timeout=6), (
            "ready lifeline did not reap worker"
        )
        record = graph.runs.get_worker(facts.invocation)
        assert record is not None
        assert record.ended_at is not None or record in graph.runs.live_workers(run_id="run-1")
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            _ = supervisor.wait(timeout=2)
        if pid and _alive(pid, token):
            os.killpg(pid, signal.SIGKILL)
        graph.close()


def test_helper_supervisor_death_gap_retains_worker_record(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path)
    marker = tmp_path / "ready.json"
    supervisor = subprocess.Popen(
        (sys.executable, "-c", _SUPERVISOR, str(graph.db_path), str(marker), str(node_id)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid = 0
    token = 0.0
    try:
        assert _until(marker.exists)
        facts = msgspec.json.decode(marker.read_bytes(), type=_SupervisorFacts)
        record = graph.runs.get_worker(facts.invocation)
        assert record is not None
        pid, token = record.pid, record.start_token
        os.kill(facts.helper, signal.SIGSTOP)
        os.kill(supervisor.pid, signal.SIGKILL)
        _ = supervisor.wait(timeout=2)
        os.kill(facts.helper, signal.SIGKILL)
        assert _alive(pid, token)
        record = graph.runs.get_worker(facts.invocation)
        assert record is not None and record.ended_at is None
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            _ = supervisor.wait(timeout=2)
        if pid and _alive(pid, token):
            os.killpg(pid, signal.SIGKILL)
        graph.close()


def test_helper_replacement_keeps_bidirectional_worker_and_exit_status(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path)
    program = (
        "import sys; "
        "print('ready', flush=True); "
        "print('first:' + sys.stdin.readline().strip(), flush=True); "
        "print('second:' + sys.stdin.readline().strip(), flush=True); "
        "sys.exit(7)"
    )
    worker = _protected(graph, node_id, program, interactive=True)
    try:
        assert worker.process.stdout is not None
        assert worker.process.stdin is not None
        assert worker.process.stdout.readline() == "ready\n"
        _ = worker.process.stdin.write("before\n")
        worker.process.stdin.flush()
        assert worker.process.stdout.readline() == "first:before\n"
        before = graph.runs.get_worker(worker.identity.invocation_id)
        assert before is not None and before.helper_pid is not None
        os.kill(before.helper_pid, signal.SIGKILL)
        assert _until(
            lambda: (
                (after := graph.runs.get_worker(worker.identity.invocation_id)) is not None
                and after.ready_generation == 1
            )
        )
        after = graph.runs.get_worker(worker.identity.invocation_id)
        assert after is not None and after.pid == before.pid == worker.process.pid
        _ = worker.process.stdin.write("after\n")
        worker.process.stdin.flush()
        assert worker.process.stdout.readline() == "second:after\n"
        assert worker.process.wait(timeout=5) == 7
        assert worker.finish(timeout=5)
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        if worker.process.poll() is None:
            os.killpg(worker.process.pid, signal.SIGKILL)
            _ = worker.process.wait(timeout=2)
        graph.close()


def test_repeated_ready_helper_deaths_stop_worker_after_three_replacements(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path)
    worker = _protected(
        graph,
        node_id,
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
    )
    try:
        for generation in range(4):
            record = graph.runs.get_worker(worker.identity.invocation_id)
            assert record is not None and record.ready_generation == generation
            assert record.helper_pid is not None
            os.kill(record.helper_pid, signal.SIGKILL)
            if generation < 3:
                assert _until(
                    lambda expected=generation + 1: (
                        (current := graph.runs.get_worker(worker.identity.invocation_id))
                        is not None
                        and current.ready_generation == expected
                    )
                )
                assert worker.process.poll() is None
        assert _until(lambda: worker.process.poll() is not None, timeout=8)
        assert worker.process.returncode == -signal.SIGKILL
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.helper_generation == 3
        assert record.ended_at is not None or record in graph.runs.live_workers(run_id="run-1")
    finally:
        if worker.process.poll() is None:
            os.killpg(worker.process.pid, signal.SIGKILL)
            _ = worker.process.wait(timeout=2)
        graph.close()


def test_normal_worker_exit_does_not_launch_replacement(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path)
    worker = _protected(graph, node_id, "raise SystemExit(9)")
    try:
        assert worker.process.wait(timeout=5) == 9
        assert worker.finish(timeout=5)
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.ready_generation == record.helper_generation == 0
        assert record.ended_at is not None
    finally:
        if worker.process.poll() is None:
            os.killpg(worker.process.pid, signal.SIGKILL)
            _ = worker.process.wait(timeout=2)
        graph.close()
