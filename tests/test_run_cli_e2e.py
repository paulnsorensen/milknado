"""End-to-end `milknado run` through the real CLI process with a fake agent binary.

Only the agent CLI is fake. The CLI entrypoint, git worktrees, merge-back, the quality
gate subprocess, and SQLite are real.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from milknado.domains.common import FlavorOverride, Gate, MilknadoConfig, NodeStatus, save_config
from milknado.domains.graph import MikadoGraph
from tests.execution_session_fixtures import init_repo
from tests.worker_fixtures import install_worker_command

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX worker shell required")

_RUN_TIMEOUT_SECONDS = 60.0
_MAX_ITERATIONS = 2
_ARTIFACT = "hello.txt"
_ARTIFACT_TEXT = "hello from the fake worker\n"

# One Claude stream-json turn: write the artifact, then claim completion.
_WORKER = f"""
import json
import os
import sys
from pathlib import Path


def emit(payload):
    print(json.dumps(payload), flush=True)


for raw in sys.stdin:
    if json.loads(raw).get("type") != "user":
        continue
    emit({{"type": "system", "subtype": "init", "session_id": "e2e", "model": "fixture"}})
    Path({_ARTIFACT!r}).write_text({_ARTIFACT_TEXT!r}, encoding="utf-8")
    with Path(os.environ["E2E_WORKER_LOG"]).open("a", encoding="utf-8") as log:
        log.write(os.getcwd() + "\\n")
    emit({{
        "type": "result",
        "subtype": "success",
        "result": "<promise>MILKNADO_NODE_COMPLETE</promise>",
        "session_id": "e2e",
    }})
    break
"""


@dataclass(frozen=True)
class _Project:
    repo: Path
    goal_id: int
    leaf_id: int
    seed: str
    worker_log: Path
    gate_log: Path


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str) -> _Project:
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
    monkeypatch.setenv("E2E_WORKER_LOG", str(tmp_path / "worker.log"))
    command = install_worker_command(
        tmp_path / "bin",
        monkeypatch,
        agent="claude",
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(worker))} "$@"\n',
    )
    gate_log = tmp_path / "gate.log"
    save_config(
        MilknadoConfig(
            execution_agent=command,
            flavors={"implement": FlavorOverride(review=False)},
            quality_gates=(Gate(command=gate.format(log=shlex.quote(str(gate_log)))),),
            worktree_pattern="milknado-wt-{node_id}-{slug}",
            concurrency_limit=1,
            max_node_attempts=1,
            max_iterations=_MAX_ITERATIONS,
            project_root=repo,
            db_path=db,
        ),
        repo / "milknado.toml",
    )
    seed = _git(repo, "rev-parse", "feature")
    return _Project(repo, goal.id, leaf.id, seed, tmp_path / "worker.log", gate_log)


def _run_cli(repo: Path) -> tuple[int, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    argv = [sys.executable, "-c", "from milknado.cli import app; app()"]
    proc = subprocess.Popen(
        [*argv, "run", "--project-root", str(repo)],
        cwd=repo,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=_RUN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        pytest.fail(f"milknado run did not exit on its own; output={output[-3000:]!r}")
    return proc.returncode, output


def _statuses(project: _Project) -> tuple[NodeStatus, NodeStatus]:
    graph = MikadoGraph(project.repo / ".milknado" / "graph.db")
    try:
        goal, leaf = graph.get_node(project.goal_id), graph.get_node(project.leaf_id)
    finally:
        graph.close()
    assert goal is not None and leaf is not None
    return goal.status, leaf.status


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_run_dispatches_gates_and_merges_worker_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, monkeypatch, f"test -f {_ARTIFACT} && pwd >> {{log}}")
    repo = project.repo

    code, output = _run_cli(repo)

    assert code == 0, output
    assert "All nodes complete. Root goal achieved." in output
    [worker_cwd] = _lines(project.worker_log)
    worktree = Path(worker_cwd)
    assert worktree != repo
    assert worktree.name.startswith(f"milknado-wt-{project.leaf_id}-")
    assert not worktree.exists()
    assert _lines(project.gate_log) == [worker_cwd]
    assert _git(repo, "rev-parse", "feature~1") == project.seed
    assert f"Milknado-Node: {project.leaf_id}" in _git(repo, "log", "-1", "--format=%B")
    assert _git(repo, "show", f"feature:{_ARTIFACT}") == _ARTIFACT_TEXT.strip()
    assert (repo / _ARTIFACT).read_text(encoding="utf-8") == _ARTIFACT_TEXT
    assert _statuses(project) == (NodeStatus.DONE, NodeStatus.DONE)


def test_run_rejects_worker_change_when_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, monkeypatch, "pwd >> {log}; false")
    repo = project.repo

    code, output = _run_cli(repo)

    assert code == 0, output
    assert "Loop ended: 1 dispatched, 0 completed, 1 failed." in output
    worker_runs = _lines(project.worker_log)
    assert len(worker_runs) == _MAX_ITERATIONS
    assert _lines(project.gate_log) == worker_runs
    assert _git(repo, "rev-parse", "feature") == project.seed
    assert not _git(repo, "ls-tree", "--name-only", "feature", "--", _ARTIFACT)
    assert not (repo / _ARTIFACT).exists()
    assert _statuses(project) == (NodeStatus.PENDING, NodeStatus.FAILED)
