"""AC-1 tracer: two projects share one host slot, so their workers never overlap."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from threading import Thread

import pytest

from milknado.app.run import build_execution_controller
from milknado.domains.common import FlavorOverride, Gate, MilknadoConfig
from milknado.domains.execution import RunLoopResult
from tests.execution_session_fixtures import build_graph, init_repo
from tests.worker_fixtures import install_worker_command

_WORKER_SECONDS = 1.0

_WORKER_SOURCE = """\
import json
import os
import sys
import time
from pathlib import Path


def mark(kind):
    with Path(os.environ["WORKER_LOG"]).open("a", encoding="utf-8") as log:
        log.write(f"{time.time()} {kind}\\n")


def emit(payload):
    print(json.dumps(payload), flush=True)


for raw in sys.stdin:
    if json.loads(raw).get("type") != "user":
        continue
    mark("start")
    emit({"type": "system", "subtype": "init", "session_id": "s", "model": "fixture"})
    time.sleep(float(os.environ["WORKER_SECONDS"]))
    Path("work.txt").write_text("done", encoding="utf-8")
    mark("end")
    emit({
        "type": "result",
        "subtype": "success",
        "result": "<promise>MILKNADO_NODE_COMPLETE</promise>",
        "session_id": "s",
    })
    break
"""


def _install_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    source = tmp_path / "sleeping_worker.py"
    _ = source.write_text(_WORKER_SOURCE, encoding="utf-8")
    log = tmp_path / "worker.log"
    monkeypatch.setenv("WORKER_LOG", str(log))
    monkeypatch.setenv("WORKER_SECONDS", str(_WORKER_SECONDS))
    command = install_worker_command(
        tmp_path / "worker-bin",
        monkeypatch,
        agent="claude",
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(source))} "$@"\n',
    )
    return command, log


def _run_project(root: Path, worker: str, limit: int, results: list[RunLoopResult]) -> None:
    root.mkdir()
    repo = init_repo(root)
    graph = build_graph(repo)
    try:
        config = MilknadoConfig(
            execution_agent=worker,
            flavors={"implement": FlavorOverride(review=False)},
            quality_gates=(Gate(command="true"),),
            worktree_pattern="milknado-wt-{node_id}-{slug}",
            concurrency_limit=1,
            host_worker_limit=limit,
            project_root=repo,
            db_path=repo / ".milknado" / "graph.db",
        )
        controller = build_execution_controller(graph, config, repo)
        results.append(controller.run(feature_branch="feature"))
    finally:
        graph.close()


def _peak_overlap(log: Path) -> int:
    events = sorted(
        (float(stamp), kind)
        for stamp, kind in (line.split() for line in log.read_text(encoding="utf-8").splitlines())
    )
    running = peak = 0
    for _, kind in events:
        running += 1 if kind == "start" else -1
        peak = max(peak, running)
    return peak


def _run_two_projects(tmp_path: Path, worker: str, limit: int) -> list[RunLoopResult]:
    results: list[RunLoopResult] = []
    threads = [
        Thread(target=_run_project, args=(tmp_path / name, worker, limit, results))
        for name in ("proj-a", "proj-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive()
    return results


def test_two_projects_never_run_workers_at_the_same_time_under_a_limit_of_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, log = _install_worker(tmp_path, monkeypatch)
    results = _run_two_projects(tmp_path, worker, limit=1)
    assert [r.completed_total for r in results] == [1, 1]
    assert log.read_text(encoding="utf-8").count("start") == 2
    assert _peak_overlap(log) == 1


def test_the_same_workers_overlap_when_the_host_limit_allows_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, log = _install_worker(tmp_path, monkeypatch)
    _ = _run_two_projects(tmp_path, worker, limit=2)
    assert _peak_overlap(log) == 2
