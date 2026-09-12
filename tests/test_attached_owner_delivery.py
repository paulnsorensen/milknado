from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread

import pytest

from milknado.app.run import ExecutionController, ExecutionSnapshot, build_execution_controller
from milknado.domains.common import FlavorOverride, Gate, MilknadoConfig
from milknado.domains.graph import MikadoGraph
from tests.execution_session_fixtures import build_graph as _build_graph
from tests.execution_session_fixtures import init_repo as _init_repo
from tests.test_execution_sessions import (
    _FOLLOW_UP,  # pyright: ignore[reportPrivateUsage]
    _READY_TEXT,  # pyright: ignore[reportPrivateUsage]
    _SESSION_ID,  # pyright: ignore[reportPrivateUsage]
    _WORKER_SOURCE,  # pyright: ignore[reportPrivateUsage]
)
from tests.worker_fixtures import install_worker_command

_ATTACHED_SOURCE = """\
from pathlib import Path
import sys

from milknado.app.watch import AttachedWatchSource, WatchSnapshotSource, graph_command_admitter
from milknado.domains.common import SessionInput
from milknado.domains.graph import MikadoGraph

repo = Path(sys.argv[1])
db_path = Path(sys.argv[2])
run_id, request_id, text = sys.argv[3:]
graph = MikadoGraph(db_path)
try:
    source = AttachedWatchSource(
        WatchSnapshotSource(repo, db_path),
        graph_command_admitter(graph),
    )
    accepted = source.session_input(
        run_id, SessionInput(action="follow_up", text=text, request_id=request_id)
    )
    print("accepted" if accepted else "rejected", flush=True)
    raise SystemExit(0 if accepted else 1)
finally:
    graph.close()
"""


def _admit_from_process(  # noqa: PLR0913
    repo: Path, db_path: Path, run_id: str, request_id: str, text: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _ATTACHED_SOURCE,
            str(repo),
            str(db_path),
            run_id,
            request_id,
            text,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _delivery_diagnostic(
    controller: ExecutionController, graph: MikadoGraph, request_id: str, path: Path
) -> str:
    snapshot = controller.snapshot()
    events = [
        (run.run_id, event.kind, event.state, event.text)
        for run in (*snapshot.active_runs, *snapshot.terminal_runs)
        for event in run.session.events
    ]
    history = [receipt.status for receipt in graph.commands.history(request_id)]
    received = path.read_text(encoding="utf-8") if path.exists() else "<missing>"
    return f"events={events!r} history={history!r} received={received!r}"


def test_attached_process_delivers_once_to_existing_owner(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    owner_graph = _build_graph(repo)
    worker_source = tmp_path / "session_worker.py"
    _ = worker_source.write_text(_WORKER_SOURCE, encoding="utf-8")
    monkeypatch.setenv("WORKER_SESSION_ID", _SESSION_ID)
    monkeypatch.setenv("WORKER_FOLLOW_UP", _FOLLOW_UP)
    received_path = tmp_path / "received.txt"
    monkeypatch.setenv("WORKER_RECEIVED_PATH", str(received_path))
    worker = install_worker_command(
        tmp_path / "worker-bin",
        monkeypatch,
        agent="claude",
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(worker_source))} "$@"\n',
    )
    config = MilknadoConfig(
        execution_agent=worker,
        flavors={"implement": FlavorOverride(review=False)},
        quality_gates=(Gate(command="true"),),
        worktree_pattern="milknado-wt-{node_id}-{slug}",
        concurrency_limit=1,
        project_root=repo,
        db_path=repo / ".milknado" / "graph.db",
    )
    controller = build_execution_controller(owner_graph, config, repo)
    ready = Event()
    delivered = Event()
    run_ids: list[str] = []

    def observe(snapshot: ExecutionSnapshot) -> None:
        for run in snapshot.active_runs:
            if any(event.text == _READY_TEXT for event in run.session.events):
                if not run_ids:
                    run_ids.append(run.run_id)
                    ready.set()
        if any(
            event.kind == "user" and event.state == "delivered" and event.text == _FOLLOW_UP
            for run in (*snapshot.active_runs, *snapshot.terminal_runs)
            for event in run.session.events
        ):
            delivered.set()

    unsubscribe = controller.subscribe(observe)
    result: list[object] = []
    runner = Thread(target=lambda: result.append(controller.run(feature_branch="feature")))
    runner.start()
    observer_graph = MikadoGraph(repo / ".milknado" / "graph.db")
    try:
        assert ready.wait(10)
        run_id = run_ids[0]
        request_id = "attached-1"
        child = _admit_from_process(
            repo, repo / ".milknado" / "graph.db", run_id, request_id, _FOLLOW_UP
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout.strip() == "accepted"
        assert delivered.wait(10), _delivery_diagnostic(
            controller, observer_graph, request_id, received_path
        )
        runner.join(10)
        assert not runner.is_alive()
        command = observer_graph.commands.command(request_id)
        assert command is not None
        history = observer_graph.commands.history(request_id)
        assert [receipt.status for receipt in history] == ["queued", "submitted", "delivered"]
        assert received_path.read_text(encoding="utf-8") == _FOLLOW_UP
        assert len([line for line in received_path.read_text(encoding="utf-8").splitlines()]) == 1
        assert len(run_ids) == 1
        assert len(result) == 1
    finally:
        unsubscribe()
        if runner.is_alive():
            controller.stop_scheduling()
            for run in controller.snapshot().active_runs:
                _ = controller.force_stop(run.run_id)
            runner.join(10)
        observer_graph.close()
        owner_graph.close()
