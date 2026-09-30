from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from milknado.adapters import GitAdapter, ProcessAdapter
from milknado.app.loop import LoopStartRequest, start_loop_run
from milknado.app.project import open_graph
from milknado.domains.common import NodeSpec, ObservationKey, pid_alive
from milknado.domains.dispatch import cancel_run, fail_stale_running_runs
from milknado.domains.graph import MikadoGraph, WorkerRecord

if os.name != "posix":
    pytest.skip("POSIX process signals are required", allow_module_level=True)


def _wait_for_worker(graph: MikadoGraph, node_id: int, marker: Path) -> WorkerRecord:
    deadline = time.monotonic() + 15
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


def _freeze_after_snapshot(graph: MikadoGraph, worker: WorkerRecord, pid: int) -> WorkerRecord:
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
    agent.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'agent.pid')!r}).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    agent.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    (root / "milknado.toml").write_text(
        '[milknado]\nagent_family = "claude"\n'
        'execution_agent = "claude -p"\nquality_gates = ["true"]\n'
        "max_iterations = 1\n"
    )
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "add", "milknado.toml"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=root,
        check=True,
    )
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


def _start(root: Path, graph: MikadoGraph, node_id: int, groups: list[int]) -> dict[str, object]:
    run = start_loop_run(graph, LoopStartRequest(node_id, None, 30, False, root))
    assert run["status"] == "running"
    pid = run["pid"]
    assert isinstance(pid, int)
    groups.append(pid)
    return run


def test_actual_runner_cancel_waits_for_worker_exit(
    running_project: tuple[Path, MikadoGraph, int, list[int]],
) -> None:
    root, graph, node_id, groups = running_project
    run = _start(root, graph, node_id, groups)
    worker = _wait_for_worker(graph, node_id, root.parent / "agent.pid")
    groups.append(worker.pgid)
    assert pid_alive(worker.pid)
    result = cancel_run(graph, GitAdapter(root), ProcessAdapter(), root, str(run["run_id"]))
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
    _freeze_after_snapshot(graph, old_worker, int(old_run["pid"]))
    os.kill(int(old_run["pid"]), signal.SIGKILL)
    _wait_for_exit(int(old_run["pid"]))
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
    _freeze_after_snapshot(graph, worker, int(run["pid"]))
    os.kill(int(run["pid"]), signal.SIGKILL)
    _wait_for_exit(int(run["pid"]))
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
    _freeze_after_snapshot(graph, worker, int(run["pid"]))
    os.kill(int(run["pid"]), signal.SIGKILL)
    _wait_for_exit(int(run["pid"]))
    result = cancel_run(graph, GitAdapter(root), ProcessAdapter(), root, str(run["run_id"]))
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
    frozen = _freeze_after_snapshot(graph, worker, int(run["pid"]))
    key = ObservationKey(
        worker.invocation_id,
        "supervisor",
        frozen.snapshot_seq + 1,
        -1,
        worker.pid,
        worker.start_token,
    )
    graph.runs.begin_worker_observation(key)
    os.kill(int(run["pid"]), signal.SIGKILL)
    _wait_for_exit(int(run["pid"]))
    with pytest.raises(RuntimeError, match="worker recovery unresolved"):
        _start(root, graph, node_id, groups)
    node = graph.get_node(node_id)
    assert node is not None and node.run_id == run["run_id"]
    assert node.worktree_path is not None and Path(node.worktree_path).exists()
    assert graph.runs.get(str(run["run_id"]))["status"] == "running"
    retained = graph.runs.get_worker(worker.invocation_id)
    assert retained is not None and retained.ended_at is None
    assert retained.observation_owner == "supervisor"
