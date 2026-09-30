from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from milknado.adapters import ProcessAdapter
from milknado.app.loop import LoopStartRequest, start_loop_run
from milknado.app.project import open_graph
from milknado.domains.common import NodeSpec, pid_alive
from milknado.domains.dispatch import fail_stale_running_runs
from milknado.domains.graph import MikadoGraph, WorkerRecord

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX helper uses passed file descriptors"
)


@pytest.fixture
def running_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, MikadoGraph, int, list[int]]]:
    root = tmp_path / "project"
    root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    agent = bin_dir / "claude"
    agent.write_text(
        f"#!{sys.executable}\n"
        "import os,time\n"
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
    owned: list[int] = []
    try:
        node = graph.add_node("actual worker", spec=NodeSpec(flavor="runner"))
        yield root, graph, node.id, owned
    finally:
        for pid in owned:
            if pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        graph.close()


def _ready_worker(graph: MikadoGraph, node_id: int) -> WorkerRecord:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        records = graph.runs.live_workers(node_id=node_id)
        if records and records[0].ready_generation == 0:
            return records[0]
        time.sleep(0.05)
    raise AssertionError("actual runner did not register a ready worker")


def _dead(pid: int) -> bool:
    return not pid_alive(pid)


def _create_death_gap(
    root: Path, graph: MikadoGraph, node_id: int, owned: list[int]
) -> tuple[str, WorkerRecord]:
    run = start_loop_run(graph, LoopStartRequest(node_id, None, 30, False, root))
    assert run["status"] == "running"
    supervisor_pid, run_id = run["pid"], run["run_id"]
    assert isinstance(supervisor_pid, int) and isinstance(run_id, str)
    owned.append(supervisor_pid)
    worker = _ready_worker(graph, node_id)
    owned.append(worker.pid)
    assert worker.helper_pid is not None
    marker = root.parent / "agent.pid"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text().strip() == str(worker.pid):
            break
        time.sleep(0.05)
    else:
        pytest.fail("fake agent did not execute after protected startup")
    assert pid_alive(worker.pid)
    os.kill(supervisor_pid, signal.SIGSTOP)
    deadline = time.monotonic() + 3
    while psutil.Process(supervisor_pid).status() != psutil.STATUS_STOPPED:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    os.kill(worker.helper_pid, signal.SIGKILL)
    deadline = time.monotonic() + 3
    while psutil.Process(worker.helper_pid).status() != psutil.STATUS_ZOMBIE:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    os.kill(supervisor_pid, signal.SIGKILL)
    deadline = time.monotonic() + 3
    while not _dead(supervisor_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _dead(supervisor_pid) and pid_alive(worker.pid)
    gap = graph.runs.get_worker(worker.invocation_id)
    assert gap is not None and gap.ended_at is None
    return run_id, worker


@pytest.mark.parametrize("corrupt_identity", [False, True], ids=["verified", "mismatched"])
def test_lifeline_supervisor_gap_recovery_verifies_before_release(
    running_project: tuple[Path, MikadoGraph, int, list[int]],
    corrupt_identity: bool,
) -> None:
    root, graph, node_id, owned = running_project
    run_id, worker = _create_death_gap(root, graph, node_id, owned)
    if corrupt_identity:
        with sqlite3.connect(graph.db_path) as db:
            db.execute(
                "UPDATE run_workers SET start_token = ? WHERE invocation_id = ?",
                (worker.start_token + 1, worker.invocation_id),
            )
    changed = fail_stale_running_runs(graph, node_id, ProcessAdapter())
    recovered = graph.runs.get_worker(worker.invocation_id)
    assert recovered is not None
    if corrupt_identity:
        assert changed == []
        assert recovered.ended_at is None
        assert pid_alive(worker.pid)
        current = graph.get_node(node_id)
        assert current is not None and current.run_id == run_id
    else:
        assert len(changed) == 1 and changed[0]["status"] == "failed"
        assert not pid_alive(worker.pid)
        assert recovered.ended_at is not None
