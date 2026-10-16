from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import TypedDict

import psutil
import pytest

from milknado.adapters import GitAdapter, ProcessAdapter
from milknado.app.loop import LoopStartRequest, start_loop_run
from milknado.app.project import open_graph
from milknado.domains.common import NodeSpec, NodeStatus, ObservationKey, pid_alive
from milknado.domains.dispatch import cancel_run, fail_stale_running_runs
from milknado.domains.graph import MikadoGraph, WorkerRecord

if os.name != "posix":
    pytest.skip("POSIX process signals are required", allow_module_level=True)


class _StartedRun(TypedDict):
    run_id: str
    pid: int


def _wait_for_worker(graph: MikadoGraph, node_id: int, marker: Path) -> WorkerRecord:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        workers = graph.runs.live_workers(node_id=node_id)
        if workers and marker.exists() and marker.read_text() == str(workers[0].pid):
            assert pid_alive(workers[0].pid)
            return workers[0]
        time.sleep(0.05)
    raise AssertionError("actual fixture agent did not stay alive")


def _wait_for_exit(pid: int) -> None:
    deadline = time.monotonic() + 5
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    status = psutil.Process(pid).status() if psutil.pid_exists(pid) else "gone"
    assert not pid_alive(pid), f"supervisor status={status}"


def freeze_after_snapshot(graph: MikadoGraph, worker: WorkerRecord, pid: int) -> WorkerRecord:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        record = graph.runs.get_worker(worker.invocation_id)
        if record is not None and record.snapshot_seq > 0 and record.observation_owner is None:
            os.kill(pid, signal.SIGSTOP)
            while psutil.Process(pid).status() != psutil.STATUS_STOPPED:
                if time.monotonic() >= deadline:
                    raise AssertionError("actual supervisor did not stop")
                time.sleep(0.01)
            frozen = graph.runs.get_worker(worker.invocation_id)
            if frozen is not None and frozen.observation_owner is None:
                try:
                    with closing(sqlite3.connect(graph.db_path, timeout=0)) as probe:
                        _ = probe.execute("BEGIN IMMEDIATE")
                        probe.rollback()
                except sqlite3.Error as exc:
                    os.kill(pid, signal.SIGCONT)
                    if not isinstance(exc, sqlite3.OperationalError) or (
                        exc.sqlite_errorcode != sqlite3.SQLITE_BUSY
                    ):
                        raise
                    time.sleep(0.01)
                    continue
                return frozen
            os.kill(pid, signal.SIGCONT)
        time.sleep(0.01)
    raise AssertionError("actual supervisor never reached a committed observation gap")


@pytest.fixture
def running_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, MikadoGraph, int, list[int]]]:
    root = tmp_path / "project"
    root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    agent = bin_dir / "claude"
    _ = agent.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                "import os, time",
                "from pathlib import Path",
                f"Path({str(tmp_path / 'agent.pid')!r}).write_text(str(os.getpid()))",
                "while True: time.sleep(1)",
            )
        )
        + "\n"
    )
    _ = agent.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    _ = (root / "milknado.toml").write_text(
        "\n".join(
            (
                "[milknado]",
                'agent_family = "claude"',
                'execution_agent = "claude -p"',
                'quality_gates = ["true"]',
                "max_iterations = 1",
            )
        )
        + "\n"
    )
    _ = subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    _ = subprocess.run(["git", "add", "milknado.toml"], cwd=root, check=True)
    git_identity = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid"]
    _ = subprocess.run([*git_identity, "commit", "-qm", "initial"], cwd=root, check=True)
    graph, _ = open_graph(root)
    groups: list[int] = []
    try:
        node = graph.add_node("actual worker", spec=NodeSpec(flavor="runner"))
        yield root, graph, node.id, groups
    finally:
        for pid in groups:
            if pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        graph.close()


def _start(root: Path, graph: MikadoGraph, node_id: int, groups: list[int]) -> _StartedRun:
    run = start_loop_run(graph, LoopStartRequest(node_id, None, 30, False, root))
    assert run["status"] == "running"
    run_id, pid = run["run_id"], run["pid"]
    assert isinstance(run_id, str) and isinstance(pid, int)
    groups.append(pid)
    return {"run_id": run_id, "pid": pid}


def _wait_for_detached_failure(graph: MikadoGraph, outer_id: str, inner_id: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        outer, inner = graph.runs.get(outer_id), graph.runs.get(inner_id)
        if outer is not None and inner is not None:
            if outer["status"] != "running" and inner["status"] != "running":
                return
        time.sleep(0.05)
    raise AssertionError("detached failure did not complete")


def test_detached_runner_reuses_one_linked_task_attempt(
    running_project: tuple[Path, MikadoGraph, int, list[int]],
) -> None:
    root, graph, node_id, groups = running_project
    config_path = root / "milknado.toml"
    _ = config_path.write_text(config_path.read_text() + "max_node_attempts = 3\n")
    run = _start(root, graph, node_id, groups)
    worker = _wait_for_worker(graph, node_id, root.parent / "agent.pid")
    groups.append(worker.pgid)
    outer = graph.runs.get(run["run_id"])
    assert outer is not None and outer["attempt_id"] is not None
    attempt_id = str(outer["attempt_id"])
    assert worker.graph_run_id is not None
    os.kill(worker.pid, signal.SIGKILL)
    _wait_for_exit(worker.pid)
    _wait_for_detached_failure(graph, run["run_id"], worker.graph_run_id)
    _wait_for_exit(run["pid"])
    graph.close()
    reopened, _ = open_graph(root)
    try:
        outer = reopened.runs.get(run["run_id"])
        inner = reopened.runs.get(worker.graph_run_id)
        attempt = reopened.task_attempts.get(attempt_id)
        current = reopened.get_node(node_id)
        evidence = reopened.runs.get_worker(worker.invocation_id)
        assert outer is not None and inner is not None
        assert (outer["status"], inner["status"]) == ("failed", "failed")
        assert (outer["attempt_id"], inner["attempt_id"]) == (attempt_id, attempt_id)
        assert attempt is not None
        assert (attempt.outcome, attempt.disposition) == ("failed", "pending")
        assert current is not None and current.status is NodeStatus.PENDING
        assert evidence is not None and evidence.ended_at is not None
        assert reopened.task_attempts.failed_count(node_id) == 1
        assert fail_stale_running_runs(reopened, node_id, ProcessAdapter()) == []
        assert reopened.task_attempts.failed_count(node_id) == 1
        replayed = (reopened.runs.get(run["run_id"]), reopened.runs.get(worker.graph_run_id))
        assert all(row is not None and row["status"] == "failed" for row in replayed)
    finally:
        reopened.close()


def test_actual_runner_cancel_waits_for_worker_exit(
    running_project: tuple[Path, MikadoGraph, int, list[int]],
) -> None:
    root, graph, node_id, groups = running_project
    run = _start(root, graph, node_id, groups)
    worker = _wait_for_worker(graph, node_id, root.parent / "agent.pid")
    groups.append(worker.pgid)
    assert pid_alive(worker.pid)
    result = cancel_run(graph, GitAdapter(root), ProcessAdapter(), root, run["run_id"])
    assert result["status"] == "failed"
    assert result["error"] == "cancelled"
    assert not pid_alive(worker.pid)
    recorded = graph.runs.get_worker(worker.invocation_id)
    assert recorded is not None and recorded.ended_at is not None


def test_start_reclaims_only_after_actual_old_worker_exits(
    running_project: tuple[Path, MikadoGraph, int, list[int]],
) -> None:
    root, graph, node_id, groups = running_project
    old_run = _start(root, graph, node_id, groups)
    old_worker = _wait_for_worker(graph, node_id, root.parent / "agent.pid")
    groups.append(old_worker.pid)
    assert pid_alive(old_worker.pid)
    _ = freeze_after_snapshot(graph, old_worker, old_run["pid"])
    os.kill(old_run["pid"], signal.SIGKILL)
    _wait_for_exit(old_run["pid"])
    replacement = _start(root, graph, node_id, groups)
    assert replacement["run_id"] != old_run["run_id"]
    assert not pid_alive(old_worker.pid)
    recorded = graph.runs.get_worker(old_worker.invocation_id)
    assert recorded is not None and recorded.ended_at is not None


def test_stale_sweep_waits_for_actual_old_worker_exit(
    running_project: tuple[Path, MikadoGraph, int, list[int]],
) -> None:
    root, graph, node_id, groups = running_project
    run = _start(root, graph, node_id, groups)
    worker = _wait_for_worker(graph, node_id, root.parent / "agent.pid")
    groups.append(worker.pid)
    assert pid_alive(worker.pid)
    _ = freeze_after_snapshot(graph, worker, run["pid"])
    os.kill(run["pid"], signal.SIGKILL)
    _wait_for_exit(run["pid"])
    changed = fail_stale_running_runs(graph, node_id, ProcessAdapter())
    assert len(changed) == 1
    assert changed[0]["status"] == "failed"
    assert not pid_alive(worker.pid)
    recorded = graph.runs.get_worker(worker.invocation_id)
    assert recorded is not None and recorded.ended_at is not None


def test_dead_owner_cancel_recovers_actual_worker(
    running_project: tuple[Path, MikadoGraph, int, list[int]],
) -> None:
    root, graph, node_id, groups = running_project
    run = _start(root, graph, node_id, groups)
    worker = _wait_for_worker(graph, node_id, root.parent / "agent.pid")
    groups.append(worker.pid)
    _ = freeze_after_snapshot(graph, worker, run["pid"])
    os.kill(run["pid"], signal.SIGKILL)
    _wait_for_exit(run["pid"])
    result = cancel_run(graph, GitAdapter(root), ProcessAdapter(), root, run["run_id"])
    assert result["status"] == "failed"
    assert result["error"] == "cancelled"
    assert not pid_alive(worker.pid)
    recorded = graph.runs.get_worker(worker.invocation_id)
    assert recorded is not None and recorded.ended_at is not None


def test_start_preserves_actual_worker_with_interrupted_observation(
    running_project: tuple[Path, MikadoGraph, int, list[int]],
) -> None:
    root, graph, node_id, groups = running_project
    run = _start(root, graph, node_id, groups)
    worker = _wait_for_worker(graph, node_id, root.parent / "agent.pid")
    groups.append(worker.pid)
    frozen = freeze_after_snapshot(graph, worker, run["pid"])
    key = ObservationKey(
        worker.invocation_id,
        "supervisor",
        frozen.snapshot_seq + 1,
        -1,
        worker.pid,
        worker.start_token,
    )
    graph.runs.begin_worker_observation(key)
    os.kill(run["pid"], signal.SIGKILL)
    _wait_for_exit(run["pid"])
    with pytest.raises(RuntimeError, match="worker recovery unresolved"):
        _ = _start(root, graph, node_id, groups)
    node = graph.get_node(node_id)
    assert node is not None and node.run_id == run["run_id"]
    assert node.worktree_path is not None and Path(node.worktree_path).exists()
    run_state = graph.runs.get(run["run_id"])
    assert run_state is not None and run_state["status"] == "running"
    retained = graph.runs.get_worker(worker.invocation_id)
    assert retained is not None and retained.ended_at is None
    assert retained.observation_owner == "supervisor"
