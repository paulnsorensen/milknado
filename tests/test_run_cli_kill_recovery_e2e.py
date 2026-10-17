"""End-to-end recovery of `milknado run` after its controller dies by SIGKILL.

Only the agent CLI is fake. The CLI process, git worktrees, merge-back, the quality gate,
and SQLite are real. The first controller is killed alone; the second `milknado run`
must recover the orphaned worker and finish the leaf exactly once.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from milknado.domains.common import (
    FlavorOverride,
    Gate,
    MilknadoConfig,
    NodeStatus,
    pid_alive,
    save_config,
)
from milknado.domains.graph import MikadoGraph
from tests.execution_session_fixtures import init_repo
from tests.worker_fixtures import install_worker_command

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX worker shell required")

_WAIT_SECONDS = 30.0
_RUN_TIMEOUT_SECONDS = 60.0
_ARTIFACT = "hello.txt"

# First attempt: record the pid, then block. Later attempts: complete normally.
_WORKER = """
import json
import os
import sys
import time
from pathlib import Path

marker = Path(os.environ["E2E_MARKER"])
release = Path(os.environ["E2E_RELEASE"])


def emit(payload):
    print(json.dumps(payload), flush=True)


for raw in sys.stdin:
    if json.loads(raw).get("type") != "user":
        continue
    emit({"type": "system", "subtype": "init", "session_id": "e2e", "model": "fixture"})
    first = not marker.exists()
    if first:
        marker.write_text(str(os.getpid()), encoding="utf-8")
        deadline = time.monotonic() + 60
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
    Path("hello.txt").write_text("hello\\n", encoding="utf-8")
    emit({
        "type": "result",
        "subtype": "success",
        "result": "<promise>MILKNADO_NODE_COMPLETE</promise>",
        "session_id": "e2e",
    })
    break
"""


@dataclass(frozen=True)
class _Project:
    repo: Path
    goal_id: int
    leaf_id: int
    marker: Path
    env: dict[str, str]

    @property
    def db(self) -> Path:
        return self.repo / ".milknado" / "graph.db"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Project]:
    repo = init_repo(tmp_path)
    db = repo / ".milknado" / "graph.db"
    db.parent.mkdir()
    graph = MikadoGraph(db)
    try:
        goal = graph.add_node("Ship the hello file")
        leaf = graph.add_node(f"Write {_ARTIFACT}", parent_id=goal.id)
    finally:
        graph.close()
    worker = tmp_path / "worker.py"
    _ = worker.write_text(_WORKER, encoding="utf-8")
    marker = tmp_path / "worker.pid"
    monkeypatch.setenv("E2E_MARKER", str(marker))
    monkeypatch.setenv("E2E_RELEASE", str(tmp_path / "release"))
    command = install_worker_command(
        tmp_path / "bin",
        monkeypatch,
        agent="claude",
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(worker))} "$@"\n',
    )
    save_config(
        MilknadoConfig(
            execution_agent=command,
            flavors={"implement": FlavorOverride(review=False)},
            quality_gates=(Gate(command=f"test -f {_ARTIFACT}"),),
            worktree_pattern="milknado-wt-{node_id}-{slug}",
            concurrency_limit=1,
            max_node_attempts=2,
            max_iterations=2,
            project_root=repo,
            db_path=db,
        ),
        repo / "milknado.toml",
    )
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    yield _Project(repo, goal.id, leaf.id, marker, env)
    if marker.exists():
        _kill(int(marker.read_text(encoding="utf-8")))


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _start_cli(project: _Project, log: Path) -> subprocess.Popen[str]:
    argv = [sys.executable, "-c", "from milknado.cli import app; app()"]
    with log.open("w", encoding="utf-8") as out:
        return subprocess.Popen(
            [*argv, "run", "--project-root", str(project.repo)],
            cwd=project.repo,
            env=project.env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )


def _wait_until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.1)
    pytest.fail(f"timed out waiting for {what}")


def _leaf_status(project: _Project) -> NodeStatus:
    graph = MikadoGraph(project.db)
    try:
        leaf = graph.get_node(project.leaf_id)
    finally:
        graph.close()
    assert leaf is not None
    return leaf.status


def _run_states(project: _Project) -> list[str]:
    graph = MikadoGraph(project.db)
    try:
        return [str(r.get("status")) for r in graph.runs.for_node(project.leaf_id)]
    finally:
        graph.close()


def _open_workers(project: _Project) -> int:
    graph = MikadoGraph(project.db)
    try:
        return len(graph.runs.live_workers(node_id=project.leaf_id))
    finally:
        graph.close()


def test_run_recovers_orphaned_worker_after_controller_sigkill(
    project: _Project, tmp_path: Path
) -> None:
    first = _start_cli(project, tmp_path / "first.log")
    try:
        _wait_until(project.marker.exists, "first worker marker")
        _wait_until(lambda: _leaf_status(project) is NodeStatus.RUNNING, "leaf RUNNING")
        orphan = int(project.marker.read_text(encoding="utf-8"))
        assert pid_alive(orphan)
        os.kill(first.pid, signal.SIGKILL)
        _ = first.wait(timeout=10)
        # The worker lifeline stops the worker once its controller is gone. The run record
        # stays running because nothing has finalized it yet.
        _wait_until(lambda: not pid_alive(orphan), "lifeline to stop the worker")
        assert _leaf_status(project) is NodeStatus.RUNNING
        assert _run_states(project) == ["running"]
    finally:
        if first.poll() is None:
            os.killpg(first.pid, signal.SIGKILL)
            _ = first.wait()

    second = _start_cli(project, tmp_path / "second.log")
    try:
        code = second.wait(timeout=_RUN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(second.pid, signal.SIGKILL)
        _ = second.wait()
        pytest.fail(f"second run hung: {(tmp_path / 'second.log').read_text()[-3000:]!r}")
    output = (tmp_path / "second.log").read_text(encoding="utf-8")

    assert code == 0, output
    assert "All nodes complete. Root goal achieved." in output
    assert not pid_alive(orphan)
    assert _leaf_status(project) is NodeStatus.DONE
    assert _run_states(project).count("failed") == 1
    assert "running" not in _run_states(project)
    assert _open_workers(project) == 0
    assert _git(project.repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    trailers = _git(project.repo, "log", "feature", "--format=%B").count(
        f"Milknado-Node: {project.leaf_id}"
    )
    assert trailers == 1
