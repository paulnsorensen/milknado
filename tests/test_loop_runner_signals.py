from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from milknado.app.project import open_graph
from milknado.domains.common import NodeSpec, pid_alive
from milknado.domains.dispatch import make_run_id, now_iso
from milknado.domains.graph import MikadoGraph, WorkerRecord


@pytest.fixture
def signal_project(
    tmp_path: Path,
) -> Iterator[tuple[Path, MikadoGraph, int, dict[str, str]]]:
    root = tmp_path / "project"
    root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    agent = bin_dir / "claude"
    agent.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        "from pathlib import Path\n"
        "Path(os.environ['FAKE_AGENT_PID_FILE']).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    agent.chmod(0o755)
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
    node = graph.add_node("signalled actual worker", spec=NodeSpec(flavor="runner"))
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_AGENT_PID_FILE": str(tmp_path / "agent.pid"),
    }
    try:
        yield root, graph, node.id, env
    finally:
        for worker in graph.runs.live_workers(node_id=node.id):
            if pid_alive(worker.pid):
                os.kill(worker.pid, signal.SIGKILL)
        graph.close()


def _wait_for_worker(graph: MikadoGraph, node_id: int) -> WorkerRecord:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        workers = graph.runs.live_workers(node_id=node_id)
        if workers:
            return workers[0]
        time.sleep(0.05)
    raise AssertionError("actual runner did not launch the fixture worker")


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_actual_runner_handles_signal_before_exiting(
    signal_project: tuple[Path, MikadoGraph, int, dict[str, str]],
    signum: signal.Signals,
) -> None:
    root, graph, node_id, env = signal_project
    graph.register_controller_master()
    run_id = make_run_id(node_id)
    graph.claim_node_for_dispatch(node_id, run_id, now=now_iso())
    log_path = root / f"{run_id}.log"
    graph.runs.start(run_id, node_id, str(log_path), now_iso(), 30)
    base_oid = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    argv = [
        sys.executable,
        "-m",
        "milknado.mcp._loop_node_runner",
        "--node-id",
        str(node_id),
        "--project-root",
        str(root),
        "--run-id",
        run_id,
        "--target-branch",
        "main",
        "--base-oid",
        base_oid,
    ]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            argv,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        worker = _wait_for_worker(graph, node_id)
        assert pid_alive(worker.pid)
        os.kill(proc.pid, signum)
        returncode = proc.wait(timeout=8)
        assert returncode == 128 + signum, (returncode, log_path.read_text())
        assert not pid_alive(worker.pid)
        record = graph.runs.get_worker(worker.invocation_id)
        assert record is not None
        if record.ended_at is None:
            node = graph.get_node(node_id)
            assert node is not None and node.run_id == run_id
            assert graph.runs.get(run_id)["status"] == "running"
            assert node.worktree_path is not None and Path(node.worktree_path).exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=2)
