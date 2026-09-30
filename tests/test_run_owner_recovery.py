"""Focused recovery tests for persisted run ownership (#215, #309)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import psutil
import pytest

from milknado.adapters.process import ProcessAdapter
from milknado.app.worker_recovery import reconcile_loop_workers
from milknado.domains.common import NodeStatus, RunResult, WorkerIdentity, WorkerOwner
from milknado.domains.dispatch import reconcile
from milknado.domains.graph import (
    MikadoGraph,
    RunFenceLostError,
    RunRecord,
    UnassociatedWorkers,
    WorkerEvidenceStore,
    open_standalone_worker_evidence,
)


@dataclass
class _RecoveryNode:
    id: int
    status: NodeStatus
    run_id: str | None
    pid: int | None


class _RecoveryGraph:
    def __init__(self, *, pid: int | None, db_path: Path, finish_succeeds: bool = True) -> None:
        self.db_path: Path = db_path
        evidence_graph = MikadoGraph(db_path)
        evidence_graph.close()
        self.node: _RecoveryNode = _RecoveryNode(
            id=1, status=NodeStatus.RUNNING, run_id="run-1", pid=pid
        )
        self.state: RunRecord = {
            "run_id": "run-1",
            "node_id": 1,
            "status": "running",
            "pid": None,
            "log_path": "",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": None,
            "timed_out": False,
            "exit_code": None,
            "error": None,
            "timeout_seconds": 300,
            "detail": None,
            "rebased": None,
        }
        self.finish_succeeds: bool = finish_succeeds
        self.runs: _RecoveryRuns = _RecoveryRuns(self)

    def get_all_nodes(self) -> list[_RecoveryNode]:
        return [self.node]

    def get_node(self, node_id: int) -> _RecoveryNode | None:
        return self.node if node_id == self.node.id else None

    def mark_terminal(self, _node_id: int, _run_id: str, status: NodeStatus) -> bool:
        self.node.status = status
        self.node.run_id = None
        return True


class _RecoveryRuns:
    def __init__(self, graph: _RecoveryGraph) -> None:
        self._graph: _RecoveryGraph = graph

    def for_node(self, node_id: int) -> list[RunRecord]:
        return [self._graph.state] if node_id == self._graph.node.id else []

    def latest_terminal(self, node_id: int, run_id: str) -> RunRecord | None:
        if node_id != self._graph.node.id or run_id != self._graph.state["run_id"]:
            return None
        return self._graph.state if self._graph.state["status"] in ("done", "failed") else None

    def finish(self, _run_id: str, result: RunResult) -> None:
        if not self._graph.finish_succeeds:
            raise RunFenceLostError("runs.finish lost its running-row fence")
        self._graph.state.update(
            status=result.status,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            ended_at=result.ended_at,
            error=result.error,
        )


def _pid_dead(_pid: int) -> bool:
    return False


def _pid_alive(_pid: int) -> bool:
    return True


def test_reconcile_orphaned_runs_finalizes_dead_coordinator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = _RecoveryGraph(pid=424242, db_path=tmp_path / "evidence.db")
    monkeypatch.setattr(reconcile, "pid_alive", _pid_dead)

    recovered = reconcile.reconcile_orphaned_runs(graph, ProcessAdapter())

    assert recovered == [graph.state]
    assert graph.state["error"] == "worker session gone"
    assert graph.node.status is NodeStatus.FAILED
    assert graph.node.run_id is None


def test_reconcile_orphaned_runs_preserves_healthy_coordinator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = _RecoveryGraph(pid=424242, db_path=tmp_path / "evidence.db")
    monkeypatch.setattr(reconcile, "pid_alive", _pid_alive)

    assert reconcile.reconcile_orphaned_runs(graph, ProcessAdapter()) == []
    assert graph.state["status"] == "running"
    assert graph.node.status is NodeStatus.RUNNING


def test_reconcile_orphaned_runs_ignores_graphs_without_node_enumeration() -> None:
    assert reconcile.reconcile_orphaned_runs(object(), ProcessAdapter()) == []


def test_dead_owner_recovery_rejects_lost_terminal_fence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = _RecoveryGraph(pid=424242, db_path=tmp_path / "evidence.db", finish_succeeds=False)
    monkeypatch.setattr(reconcile, "pid_alive", _pid_dead)

    with pytest.raises(RunFenceLostError, match="running-row fence"):
        _ = reconcile.fail_stale_running_runs(graph, 1, ProcessAdapter())


def test_blocked_cleanup_does_not_hold_process_exit(tmp_path: Path) -> None:
    script = """
import os, signal, subprocess, sys, time
from pathlib import Path
import psutil
from milknado.adapters.process import ProcessAdapter
from milknado.domains.common import WorkerIdentity, WorkerOwner
from milknado.domains.dispatch.reap import ReapRequest, reap_orphaned_workers
from milknado.domains.graph import MikadoGraph, NodeWorkers
class Blocked(ProcessAdapter):
    def terminate_worker(self, worker, retained, deadline):
        time.sleep(30)
proc = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True
)
graph = MikadoGraph(Path(sys.argv[1]) / 'graph.db')
try:
    node = graph.add_node('bounded')
    graph.claim_node(node.id, 'run-1', now='2026-01-01T00:00:00+00:00', pid=999999)
    graph.runs.start('run-1', node.id, 'worker.log', '2026-01-01T00:00:00+00:00', None)
    graph.runs.record_worker(
        WorkerOwner('run-1', 999999, 123.5, 'run-1', node.id),
        WorkerIdentity('inv-1', proc.pid, proc.pid, psutil.Process(proc.pid).create_time()),
    )
    start = time.monotonic()
    assert not reap_orphaned_workers(
        graph, Blocked(), ReapRequest(NodeWorkers(node.id), deadline=start + .2)
    )
    assert time.monotonic() - start < 1 and graph.runs.live_workers(node_id=node.id)
finally:
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=3)
    graph.close()
print('bounded', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=8
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bounded"


def _detached_worker() -> int:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',"
            + "'import time; time.sleep(60)'],start_new_session=True,"
            + "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(p.pid)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


@pytest.mark.parametrize("standalone", [False, True])
def test_app_preflight_recovers_unassociated_worker(tmp_path: Path, standalone: bool) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    pid = _detached_worker()
    try:
        token = psutil.Process(pid).create_time()
        if standalone:
            store_context = open_standalone_worker_evidence()
        else:
            store_context = WorkerEvidenceStore(graph.db_path)
        with store_context as store:
            path = store.db_path
            store.record_worker(
                WorkerOwner("runtime-1", 2**31 - 1, 123.5),
                WorkerIdentity("inv-1", pid, pid, token),
            )
        reconcile_loop_workers(graph)
        with WorkerEvidenceStore(path) as store:
            assert store.live_workers(UnassociatedWorkers()) == ()
        assert not psutil.pid_exists(pid)
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        graph.close()


def test_app_preflight_skips_live_supervisor(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    pid = _detached_worker()
    try:
        with WorkerEvidenceStore(graph.db_path) as store:
            store.record_worker(
                WorkerOwner("runtime-1", os.getpid(), psutil.Process().create_time()),
                WorkerIdentity("inv-1", pid, pid, psutil.Process(pid).create_time()),
            )
        reconcile_loop_workers(graph)
        with WorkerEvidenceStore(graph.db_path) as store:
            assert len(store.live_workers(UnassociatedWorkers())) == 1
        assert psutil.pid_exists(pid)
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        graph.close()


def test_app_preflight_retains_mismatched_supervisor(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    pid = _detached_worker()
    try:
        with WorkerEvidenceStore(graph.db_path) as store:
            store.record_worker(
                WorkerOwner("runtime-1", os.getpid(), psutil.Process().create_time() + 100),
                WorkerIdentity("inv-1", pid, pid, psutil.Process(pid).create_time()),
            )
        reconcile_loop_workers(graph)
        with WorkerEvidenceStore(graph.db_path) as store:
            record = store.get("inv-1")
            assert record is not None and record.ended_at is None
        assert psutil.pid_exists(pid)
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        graph.close()
