from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import psutil
import pytest

from milknado.adapters import GitAdapter
from milknado.adapters.process import ProcessAdapter
from milknado.app.loop import LoopStartRequest, _claim_loop  # pyright: ignore[reportPrivateUsage]
from milknado.domains.common import NodeStatus, ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.dispatch.cancel import cancel_run
from milknado.domains.dispatch.reap import ReapRequest, reap_orphaned_workers
from milknado.domains.dispatch.reconcile import fail_stale_running_runs
from milknado.domains.graph import MikadoGraph, NodeWorkers, RunWorkers


@pytest.fixture
def worker() -> Iterator[int]:
    launcher = subprocess.Popen(
        [
            sys.executable, "-c",
            "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',"
            "'import time; time.sleep(60)'],start_new_session=True,"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(p.pid,flush=True)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert launcher.stdout is not None
    pid = int(launcher.stdout.readline())
    assert launcher.wait(timeout=3) == 0
    try:
        yield pid
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)


def _graph(tmp_path: Path, worker: int, token: float) -> MikadoGraph:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("recover")
    assert graph.claim_node(node.id, "run-1", now="2026-01-01T00:00:00+00:00", pid=2**31 - 1)
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    graph.runs.record_worker(
        WorkerOwner("run-1", 2**31 - 1, 123.5, "run-1", node.id),
        WorkerIdentity("inv-1", worker, worker, token),
    )
    return graph


def test_reap_confirms_worker_exit_before_record_closure(tmp_path: Path, worker: int) -> None:
    graph = _graph(tmp_path, worker, psutil.Process(worker).create_time())
    try:
        node = graph.get_all_nodes()[0]
        assert reap_orphaned_workers(
            graph, ProcessAdapter(), ReapRequest(NodeWorkers(node.id), deadline=time.monotonic() + 3)
        )
        assert graph.runs.live_workers(node_id=node.id) == ()
        assert not psutil.pid_exists(worker)
        assert graph.try_reclaim(node.id, now="2026-01-01T00:01:00+00:00")
    finally:
        graph.close()


def test_reap_refuses_mismatched_leader_without_signal(tmp_path: Path, worker: int) -> None:
    graph = _graph(tmp_path, worker, psutil.Process(worker).create_time() - 100)
    try:
        node = graph.get_all_nodes()[0]
        assert not reap_orphaned_workers(
            graph, ProcessAdapter(), ReapRequest(NodeWorkers(node.id), deadline=time.monotonic() + 0.3)
        )
        assert psutil.pid_exists(worker)
        assert graph.runs.live_workers(node_id=node.id)
        assert graph.try_reclaim(node.id, now="2026-01-01T00:01:00+00:00") is False
        assert graph.get_node(node.id).status is NodeStatus.RUNNING
    finally:
        graph.close()


@contextmanager
def _orphan_descendant(separate_session: bool) -> Iterator[tuple[int, float, int, float, int]]:
    leader = subprocess.Popen(
        [
            sys.executable, "-c",
            "import subprocess,sys; sys.stdin.readline(); "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "start_new_session=bool(int(sys.argv[1])),stdout=subprocess.DEVNULL); "
            "print(p.pid,flush=True)",
            str(int(separate_session)),
        ],
        start_new_session=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    token = psutil.Process(leader.pid).create_time()
    assert leader.stdin is not None and leader.stdout is not None
    leader.stdin.write("\n")
    leader.stdin.flush()
    child = int(leader.stdout.readline())
    assert leader.wait(timeout=3) == 0
    child_token = psutil.Process(child).create_time()
    child_group = os.getpgid(child)
    try:
        yield leader.pid, token, child, child_token, child_group
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(child_group, signal.SIGKILL)


def test_dead_leader_does_not_authorize_historical_group_signal(tmp_path: Path) -> None:
    with _orphan_descendant(False) as (leader, token, child, _child_token, _group):
        graph = _graph(tmp_path, leader, token)
        try:
            node = graph.get_all_nodes()[0]
            assert not reap_orphaned_workers(
                graph, ProcessAdapter(), ReapRequest(NodeWorkers(node.id), deadline=time.monotonic() + 0.2)
            )
            assert psutil.pid_exists(child)
            assert graph.runs.live_workers(node_id=node.id)
            assert graph.try_reclaim(node.id, now="2026-01-01T00:01:00+00:00") is False
        finally:
            graph.close()


def test_reap_uses_retained_reparented_setsid_identity(tmp_path: Path) -> None:
    with _orphan_descendant(True) as (leader, token, child, child_token, child_group):
        graph = _graph(tmp_path, leader, token)
        try:
            graph.runs.begin_worker_observation(
                ObservationKey("inv-1", "supervisor", 1, -1, leader, token)
            )
            graph.runs.commit_worker_observation(
                ObservationKey("inv-1", "supervisor", 1, -1, leader, token),
                ((child, child_token, child_group),),
            )
            assert reap_orphaned_workers(
                graph, ProcessAdapter(), ReapRequest(RunWorkers("run-1"), deadline=time.monotonic() + 3)
            )
            assert not psutil.pid_exists(child)
            assert graph.runs.live_workers(run_id="run-1") == ()
        finally:
            graph.close()


def test_interrupted_observation_preserves_running_owner(
    tmp_path: Path, worker: int, caplog: pytest.LogCaptureFixture
) -> None:
    token = psutil.Process(worker).create_time()
    graph = _graph(tmp_path, worker, token)
    try:
        graph.runs.begin_worker_observation(
            ObservationKey("inv-1", "supervisor", 1, -1, worker, token)
        )
        assert not reap_orphaned_workers(graph, ProcessAdapter(), ReapRequest(NodeWorkers(1)))
        assert "interrupted observation" in caplog.text
        assert psutil.pid_exists(worker)
        assert graph.runs.live_workers(node_id=1)
        assert graph.get_node(1).status is NodeStatus.RUNNING
    finally:
        graph.close()


def test_stale_sweep_reaps_before_terminal_run(tmp_path: Path, worker: int) -> None:
    graph = _graph(tmp_path, worker, psutil.Process(worker).create_time())
    try:
        flipped = fail_stale_running_runs(graph, 1, ProcessAdapter())
        assert len(flipped) == 1 and flipped[0]["status"] == "failed"
        assert graph.runs.live_workers(node_id=1) == ()
        assert not psutil.pid_exists(worker)
    finally:
        graph.close()


def test_dead_owner_cancel_reaps_before_finalization(tmp_path: Path, worker: int) -> None:
    graph = _graph(tmp_path, worker, psutil.Process(worker).create_time())
    try:
        result = cancel_run(graph, GitAdapter(tmp_path), ProcessAdapter(), tmp_path, "run-1")
        assert result["status"] == "failed"
        assert result["worktree_preserved"] is None
        assert graph.runs.live_workers(node_id=1) == ()
        assert not psutil.pid_exists(worker)
        assert graph.get_node(1).status is NodeStatus.FAILED
    finally:
        graph.close()


def test_dead_owner_cancel_preserves_unresolved_worker(tmp_path: Path, worker: int) -> None:
    graph = _graph(tmp_path, worker, psutil.Process(worker).create_time() - 100)
    try:
        with pytest.raises(RuntimeError, match="worker recovery unresolved"):
            cancel_run(graph, GitAdapter(tmp_path), ProcessAdapter(), tmp_path, "run-1")
        assert graph.runs.get("run-1")["status"] == "running"
        assert graph.runs.live_workers(node_id=1)
        assert graph.get_node(1).status is NodeStatus.RUNNING
        assert psutil.pid_exists(worker)
    finally:
        graph.close()


def test_pid_cancel_reaps_worker_after_supervisor(tmp_path: Path, worker: int) -> None:
    with _orphan_descendant(True) as (_leader, _token, supervisor, _child_token, _group):
        graph = _graph(tmp_path, worker, psutil.Process(worker).create_time())
        try:
            graph.runs.set_pid("run-1", supervisor)
            graph.set_pid(1, "run-1", supervisor)
            result = cancel_run(graph, GitAdapter(tmp_path), ProcessAdapter(), tmp_path, "run-1")
            assert result["status"] == "failed"
            assert not psutil.pid_exists(supervisor)
            assert not psutil.pid_exists(worker)
            assert graph.runs.live_workers(node_id=1) == ()
            assert graph.get_node(1).status is NodeStatus.FAILED
        finally:
            graph.close()


def test_dispatch_refuses_reclaim_with_unresolved_identity(tmp_path: Path, worker: int) -> None:
    graph = _graph(tmp_path, worker, psutil.Process(worker).create_time() - 100)
    request = LoopStartRequest(1, None, 30, False, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="worker recovery unresolved"):
            _claim_loop(graph, GitAdapter(tmp_path), request)
        assert graph.get_node(1).status is NodeStatus.RUNNING
        assert graph.runs.get("run-1")["status"] == "running"
        assert graph.runs.live_workers(node_id=1)
        assert psutil.pid_exists(worker)
    finally:
        graph.close()


def test_dispatch_reclaims_only_after_worker_exit(tmp_path: Path, worker: int) -> None:
    graph = _graph(tmp_path, worker, psutil.Process(worker).create_time())
    request = LoopStartRequest(1, None, 30, False, tmp_path)
    for command in (
        ("git", "init", "-b", "main", str(tmp_path)),
        ("git", "-C", str(tmp_path), "config", "user.email", "test@example.com"),
        ("git", "-C", str(tmp_path), "config", "user.name", "Test"),
        ("git", "-C", str(tmp_path), "commit", "--allow-empty", "-m", "base"),
    ):
        subprocess.run(command, check=True, capture_output=True)
    try:
        claim = _claim_loop(graph, GitAdapter(tmp_path), request)
        assert claim.run_id != "run-1"
        assert graph.runs.live_workers(node_id=1) == ()
        assert not psutil.pid_exists(worker)
        assert graph.get_node(1).run_id == claim.run_id
    finally:
        graph.close()


def test_locked_evidence_database_expires_without_signal(tmp_path: Path, worker: int) -> None:
    graph = _graph(tmp_path, worker, psutil.Process(worker).create_time())
    blocker = sqlite3.connect(graph.db_path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        assert not reap_orphaned_workers(
            graph, ProcessAdapter(), ReapRequest(NodeWorkers(1), deadline=started + 0.3)
        )
        assert time.monotonic() - started < 1
        assert psutil.pid_exists(worker)
        assert graph.runs.live_workers(node_id=1)
        assert graph.get_node(1).status is NodeStatus.RUNNING
    finally:
        blocker.rollback()
        blocker.close()
        graph.close()
