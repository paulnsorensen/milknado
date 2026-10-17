"""End-to-end `milknado run` over multi-step graphs with a fake agent binary.

Only the agent CLI is fake. The CLI entrypoint, scheduler, git worktrees, merge-back, the
quality gate subprocess, and SQLite are real.
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from milknado.domains.common import FlavorOverride, Gate, MilknadoConfig, NodeStatus, save_config
from milknado.domains.graph import MikadoGraph
from tests.execution_session_fixtures import init_repo
from tests.run_cli_fixtures import git, lines, run_cli
from tests.worker_fixtures import install_worker_command

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX worker shell required")

# The worker reads its node id from its worktree name and its behavior from spec.json.
# Every start and end lands in events.log, outside the repo, in real order.
_WORKER = """
import json
import os
import sys
import time
from pathlib import Path

events = Path(os.environ["E2E_EVENTS"])
node = Path.cwd().name.split("-")[2]
spec = json.loads((events / "spec.json").read_text(encoding="utf-8"))[node]


def emit(payload):
    print(json.dumps(payload), flush=True)


def record(kind, detail=""):
    with (events / "events.log").open("a", encoding="utf-8") as log:
        log.write("\\t".join([node, kind, detail]) + "\\n")


def await_peer(peer):
    marker = events / f"started-{peer}"
    deadline = time.monotonic() + 20
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    record("peer", "seen" if marker.exists() else "missing")


for raw in sys.stdin:
    if json.loads(raw).get("type") != "user":
        continue
    emit({"type": "system", "subtype": "init", "session_id": "e2e", "model": "fixture"})
    record("start", " ".join(sorted(os.listdir("."))))
    with (events / "attempts.log").open("a", encoding="utf-8") as attempts:
        attempts.write(node + "\\n")
    (events / f"started-{node}").touch()
    if "peer" in spec:
        await_peer(spec["peer"])
    Path(spec["file"]).write_text(node + "\\n", encoding="utf-8")
    record("end")
    emit({
        "type": "result",
        "subtype": "success",
        "result": "<promise>MILKNADO_NODE_COMPLETE</promise>",
        "session_id": "e2e",
    })
    break
"""


@dataclass(frozen=True)
class _Event:
    node: str
    kind: str
    detail: str


@dataclass(frozen=True)
class _Run:
    repo: Path
    ids: dict[str, int]
    events: Path
    output: str
    code: int

    def log(self) -> list[_Event]:
        return [_Event(*line.split("\t")) for line in lines(self.events / "events.log")]

    def position(self, name: str, kind: str) -> int:
        node = str(self.ids[name])
        return next(i for i, e in enumerate(self.log()) if e.node == node and e.kind == kind)

    def statuses(self) -> dict[str, NodeStatus]:
        graph = MikadoGraph(self.repo / ".milknado" / "graph.db")
        try:
            nodes = {name: graph.get_node(node_id) for name, node_id in self.ids.items()}
        finally:
            graph.close()
        assert all(node is not None for node in nodes.values())
        return {name: node.status for name, node in nodes.items() if node is not None}


@dataclass(frozen=True)
class _Plan:
    tree: tuple[tuple[str, str | None], ...]
    files: dict[str, str]
    peers: dict[str, str]
    gate: str = "true"
    concurrency: int = 1
    attempts: int = 1
    iterations: int = 1


def _build_graph(db: Path, tree: tuple[tuple[str, str | None], ...]) -> dict[str, int]:
    db.parent.mkdir()
    graph = MikadoGraph(db)
    ids: dict[str, int] = {}
    try:
        for name, parent in tree:
            parent_id = None if parent is None else ids[parent]
            ids[name] = graph.add_node(name, parent_id=parent_id).id
    finally:
        graph.close()
    return ids


def _spec(plan: _Plan, ids: dict[str, int]) -> dict[str, dict[str, str]]:
    spec = {str(ids[name]): {"file": file} for name, file in plan.files.items()}
    for name, peer in plan.peers.items():
        spec[str(ids[name])]["peer"] = str(ids[peer])
    return spec


def _execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: _Plan) -> _Run:
    repo = init_repo(tmp_path)
    db = repo / ".milknado" / "graph.db"
    ids = _build_graph(db, plan.tree)
    events = tmp_path / "events"
    events.mkdir()
    _ = (events / "spec.json").write_text(json.dumps(_spec(plan, ids)), encoding="utf-8")
    worker = tmp_path / "worker.py"
    _ = worker.write_text(_WORKER, encoding="utf-8")
    monkeypatch.setenv("E2E_EVENTS", str(events))
    command = install_worker_command(
        tmp_path / "bin",
        monkeypatch,
        agent="claude",
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(worker))} "$@"\n',
    )
    gate = plan.gate.format(attempts=shlex.quote(str(events / "attempts.log")))
    save_config(
        MilknadoConfig(
            execution_agent=command,
            flavors={"implement": FlavorOverride(review=False)},
            quality_gates=(Gate(command=gate),),
            worktree_pattern="milknado-wt-{node_id}-{slug}",
            concurrency_limit=plan.concurrency,
            max_node_attempts=plan.attempts,
            max_iterations=plan.iterations,
            project_root=repo,
            db_path=db,
        ),
        repo / "milknado.toml",
    )
    code, output = run_cli(repo)
    return _Run(repo, ids, events, output, code)


def _assert_goal_achieved(run: _Run) -> None:
    assert run.code == 0, run.output
    assert "All nodes complete. Root goal achieved." in run.output
    assert set(run.statuses().values()) == {NodeStatus.DONE}


def _assert_merged(run: _Run, name: str, file: str) -> None:
    assert git(run.repo, "show", f"feature:{file}") == str(run.ids[name])
    trailers = git(run.repo, "log", "feature", "--format=%B", "--", file)
    assert f"Milknado-Node: {run.ids[name]}" in trailers


def test_prerequisite_merges_before_dependent_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _Plan(
        tree=(("goal", None), ("A", "goal"), ("B", "A")),
        files={"A": "a.txt", "B": "b.txt"},
        peers={},
        concurrency=2,
    )

    run = _execute(tmp_path, monkeypatch, plan)

    _assert_goal_achieved(run)
    assert run.position("B", "end") < run.position("A", "start")
    a_start = run.log()[run.position("A", "start")]
    assert "b.txt" in a_start.detail.split()
    assert "a.txt" not in a_start.detail.split()
    _assert_merged(run, "B", "b.txt")
    _assert_merged(run, "A", "a.txt")
    commits = git(run.repo, "log", "--reverse", "--format=%B", "feature")
    assert commits.index(f"Milknado-Node: {run.ids['B']}") < commits.index(
        f"Milknado-Node: {run.ids['A']}"
    )


def test_independent_siblings_run_in_parallel_and_both_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _Plan(
        tree=(("goal", None), ("C", "goal"), ("D", "goal")),
        files={"C": "c.txt", "D": "d.txt"},
        peers={"C": "D", "D": "C"},
        concurrency=2,
    )

    run = _execute(tmp_path, monkeypatch, plan)

    _assert_goal_achieved(run)
    # Each worker blocks until its sibling has started, so both saw the other running.
    peers = [e for e in run.log() if e.kind == "peer"]
    assert [e.detail for e in peers] == ["seen", "seen"]
    _assert_merged(run, "C", "c.txt")
    _assert_merged(run, "D", "d.txt")


def test_failed_first_attempt_is_requeued_and_second_attempt_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _Plan(
        tree=(("goal", None), ("E", "goal")),
        files={"E": "e.txt"},
        peers={},
        gate='test "$(wc -l < {attempts})" -ge 2',
        attempts=2,
    )

    run = _execute(tmp_path, monkeypatch, plan)

    _assert_goal_achieved(run)
    assert lines(run.events / "attempts.log") == [str(run.ids["E"])] * 2
    assert [e.kind for e in run.log()] == ["start", "end"] * 2
    _assert_merged(run, "E", "e.txt")
    assert git(run.repo, "log", "feature", "--format=%B").count("Milknado-Node:") == 1
