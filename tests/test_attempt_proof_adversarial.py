"""A live pre-exec wrapper cannot prove target command execution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.adapters._loop_worker_release import LoopWorkerRelease
from milknado.domains.common import NodeStatus, RunResult, WorkerOwner
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, StartEvidence, WorkerProcess, spawn_gated

pytestmark = pytest.mark.real_process


def _release_worker(worker: WorkerProcess, context: ProtectionContext) -> None:
    context.release.release_worker(worker, context.owner)


_NOW = "2026-01-01T00:00:00+00:00"


def _pre_exec_fault(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    ready = tmp_path / "wrapper-in-atexit"
    release = tmp_path / "release-wrapper"
    _ = (tmp_path / "sitecustomize.py").write_text(
        "".join(
            (
                "import atexit, os, pathlib, time\n",
                "def fail_exec(*args):\n",
                "    def hold_wrapper():\n",
                "        pathlib.Path(os.environ['PRESS_READY']).touch()\n",
                "        deadline = time.monotonic() + 5\n",
                "        while not pathlib.Path(os.environ['PRESS_RELEASE']).exists()",
                " and time.monotonic() < deadline:\n",
                "            time.sleep(0.01)\n",
                "    atexit.register(hold_wrapper)\n",
                "    raise ValueError('pre-exec wrapper fault')\n",
                "os.execvpe = fail_exec\n",
            )
        )
    )
    env = os.environ | {
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "PRESS_READY": str(ready),
        "PRESS_RELEASE": str(release),
    }
    return env, ready, release


def _attempt_context(path: Path) -> tuple[MikadoGraph, int, ProtectionContext]:
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "worker", 3))
    graph.runs.start("worker", node.id, "worker.log", _NOW, None, attempt_id="attempt")
    supervisor = psutil.Process()
    owner = WorkerOwner("worker", supervisor.pid, supervisor.create_time(), "worker", node.id)
    context = ProtectionContext(
        LoopWorkerEvidence(path), owner, path, release=LoopWorkerRelease(path)
    )
    return graph, node.id, context


def _failed_result() -> RunResult:
    return RunResult(
        status="failed",
        exit_code=72,
        timed_out=False,
        ended_at=_NOW,
        error="pre-exec wrapper fault",
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec-status channel")
def test_live_pre_exec_wrapper_does_not_resolve_unknown_start(tmp_path: Path) -> None:
    env, ready, release = _pre_exec_fault(tmp_path)
    graph, node_id, context = _attempt_context(tmp_path / "graph.db")
    worker = spawn_gated(
        SpawnOptions(
            command=(sys.executable, "-c", "pass"),
            cwd=None,
            env=env,
            text=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    )
    try:
        _release_worker(worker, context)
        assert ready.exists()
        assert worker.process.poll() is None
        attempt = graph.task_attempts.get("attempt")
        assert attempt is not None
        disposition = graph.task_attempts.fail(
            FinalizeTaskFailure(node_id, "attempt", "owner", "worker", _failed_result())
        )
        node = graph.get_node(node_id)
        assert node is not None
        assert (
            worker.start_evidence,
            attempt.start_state,
            disposition.status,
            disposition.failed_count,
            node.status,
        ) == (StartEvidence.UNKNOWN, "released", "unknown_start", 0, NodeStatus.RUNNING)
    finally:
        release.touch()
        worker.close_gate()
        assert worker.process.wait(timeout=3) == 72
        graph.close()
