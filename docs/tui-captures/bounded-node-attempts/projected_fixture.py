from __future__ import annotations

import argparse
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fixture import FixtureSource

from milknado.app.run import ExecutionController, ExecutionSnapshot
from milknado.app.run_tui import ExecutionApp
from milknado.app.watch_tui import WatchApp
from milknado.domains.common import MilknadoConfig, RunResult, SessionView
from milknado.domains.execution import RunLoop
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


class FixtureLoop:
    def __init__(self, output: tuple[str, ...]) -> None:
        self.output = output

    def get_run(self, run_id: str) -> None:
        del run_id
        return None

    def get_run_output_tail(self, run_id: str, limit: int) -> tuple[str, ...]:
        del run_id
        return self.output[-limit:]

    def get_run_guidance(self, run_id: str) -> tuple[str, ...]:
        del run_id
        return ()

    def get_run_session(self, run_id: str) -> SessionView:
        del run_id
        return SessionView()


def seed_graph(graph: MikadoGraph, state: str) -> tuple[int, str, tuple[str, ...]]:
    retry = state == "retry-error"
    root = graph.add_node("Bounded node attempts")
    task = graph.add_node(
        "Retry failed node after worker exit" if retry else "Unknown-start held node",
        parent_id=root.id,
    )
    assert graph.claim_node(task.id, "run-1", now=_NOW)
    graph.task_attempts.start(StartTaskAttempt(task.id, "attempt-1", "run-1", max_node_attempts=3))
    graph.runs.start("run-1", task.id, "run.log", _NOW, None, attempt_id="attempt-1")
    if retry:
        graph.task_attempts.confirm_execution("attempt-1", "run-1", "run-1")
    else:
        graph.task_attempts.record_release("attempt-1", "run-1", "run-1")
    error = "worker exited with code 1" if retry else "worker start outcome unknown"
    result = RunResult(status="failed", exit_code=1, timed_out=False, ended_at=_NOW, error=error)
    disposition = graph.task_attempts.fail(
        FinalizeTaskFailure(task.id, "attempt-1", "run-1", "run-1", 3, result)
    )
    if not retry:
        assert disposition.status == "unknown_start" and disposition.failed_count == 0
        return task.id, "run-1", (error, "retry held")
    assert disposition.failed_count == 1
    assert graph.claim_node(task.id, "run-2", now=_NOW)
    graph.task_attempts.start(StartTaskAttempt(task.id, "attempt-2", "run-2", max_node_attempts=3))
    graph.runs.start("run-2", task.id, "run.log", _NOW, None, attempt_id="attempt-2")
    return task.id, "run-2", (error, "retry scheduled")


def projected_fixture(path: Path, state: str) -> tuple[MikadoGraph, ExecutionSnapshot]:
    graph = MikadoGraph(path)
    node_id, owner, output = seed_graph(graph, state)
    loop = RunLoop(
        executor=SimpleNamespace(),
        graph=graph,
        loop=FixtureLoop(output),
        config=MilknadoConfig(max_node_attempts=3, dispatch_max_retries=1),
    )
    loop._scheduler.admit_run(owner, node_id, time.monotonic() - 12)
    loop._logs.extend(output)
    current = ExecutionController._project_snapshot(loop.state(), graph)
    assert current.active_runs[0].attempt == (2 if state == "retry-error" else 1)
    assert current.active_runs[0].max_attempts == 3
    return graph, current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("surface", choices=("run", "watch"))
    parser.add_argument("state", choices=("retry-error", "unknown-start"))
    args = parser.parse_args()
    with TemporaryDirectory(prefix="issue495-tui-") as directory:
        graph, current = projected_fixture(Path(directory) / "graph.db", args.state)
        try:
            source = FixtureSource(current)
            app = (
                ExecutionApp(source)
                if args.surface == "run"
                else WatchApp(source, poll_interval=60)
            )
            app.run()
        finally:
            graph.close()


if __name__ == "__main__":
    main()
