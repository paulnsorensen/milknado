from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest

from milknado.adapters import LoopAdapter
from milknado.app.run import ExecutionController, ExecutionSnapshot, build_execution_controller
from milknado.domains.common import FlavorOverride, Gate, MilknadoConfig, SessionInput
from milknado.domains.execution import RunLoopResult
from milknado.domains.graph import MikadoGraph
from tests.worker_fixtures import install_worker_command

_SESSION_ID = "fixture-session-42"
_FOLLOW_UP = "human follow-up"
_REQUEST_ID = "human-1"
_READY_TEXT = "Claude session started (fixture)"

_WORKER_SOURCE = """\
import json
import os
import sys
from pathlib import Path


def emit(payload):
    print(json.dumps(payload), flush=True)


user_count = 0
for raw in sys.stdin:
    payload = json.loads(raw)
    if payload.get("type") != "user":
        continue
    message = payload.get("message", {})
    text = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(text, str):
        continue
    user_count += 1
    session_id = os.environ["WORKER_SESSION_ID"]
    if user_count == 1:
        emit({
            "type": "system",
            "subtype": "init",
            "session_id": session_id,
            "model": "fixture",
        })
    elif text == os.environ["WORKER_FOLLOW_UP"]:
        Path(os.environ["WORKER_RECEIVED_PATH"]).write_text(text, encoding="utf-8")
        Path("guidance.txt").write_text(text, encoding="utf-8")
        emit({
            "type": "user",
            "message": {"role": "user", "content": text},
            "session_id": session_id,
        })
        emit({
            "type": "result",
            "subtype": "success",
            "result": "initial response",
            "session_id": session_id,
        })
        emit({
            "type": "result",
            "subtype": "success",
            "result": "<promise>MILKNADO_NODE_COMPLETE</promise>",
            "session_id": session_id,
        })
        break
"""


def _git(repo: Path, *args: str) -> None:
    _ = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "feature")
    _git(repo, "config", "user.email", "test@milknado.test")
    _git(repo, "config", "user.name", "Milknado Test")
    _ = (repo / "README.md").write_text("# session test\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _build_graph(repo: Path) -> MikadoGraph:
    db_path = repo / ".milknado" / "graph.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    graph = MikadoGraph(db_path)
    root = graph.add_node("Interactive session goal")
    _ = graph.add_node("Accept human guidance", parent_id=root.id)
    return graph


def _install_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    source = tmp_path / "session_worker.py"
    _ = source.write_text(_WORKER_SOURCE, encoding="utf-8")
    monkeypatch.setenv("WORKER_SESSION_ID", _SESSION_ID)
    monkeypatch.setenv("WORKER_FOLLOW_UP", _FOLLOW_UP)
    monkeypatch.setenv("WORKER_RECEIVED_PATH", str(tmp_path / "received.txt"))
    return install_worker_command(
        tmp_path / "worker-bin",
        monkeypatch,
        agent="claude",
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(source))} "$@"\n',
    )


def _run_controller_session(
    controller: ExecutionController,
) -> tuple[RunLoopResult, ExecutionSnapshot, str]:
    ready, receipt = Event(), Event()
    ready_run_id: list[str] = []

    def observe(snapshot: ExecutionSnapshot) -> None:
        runs = (*snapshot.active_runs, *snapshot.terminal_runs)
        for run in snapshot.active_runs:
            if any(event.text == _READY_TEXT for event in run.session.events):
                if not ready_run_id:
                    ready_run_id.append(run.run_id)
                    ready.set()
                break
        if any(
            event.kind == "user" and event.state == "delivered" and event.text == _FOLLOW_UP
            for run in runs
            for event in run.session.events
        ):
            receipt.set()

    unsubscribe = controller.subscribe(observe)
    result: list[RunLoopResult] = []
    runner = Thread(target=lambda: result.append(controller.run(feature_branch="feature")))
    runner.start()
    try:
        assert ready.wait(timeout=10)
        run_id = ready_run_id[0]
        assert (
            controller.session_input(
                run_id, SessionInput(action="follow_up", text=_FOLLOW_UP, request_id=_REQUEST_ID)
            )
            is True
        )
        assert receipt.wait(timeout=10)
        runner.join(timeout=10)
        assert not runner.is_alive()
        assert len(result) == 1
        return result[0], controller.snapshot(), run_id
    finally:
        unsubscribe()
        if runner.is_alive():
            controller.stop_scheduling()
            for run in controller.snapshot().active_runs:
                _ = controller.force_stop(run.run_id)
            runner.join(timeout=10)
            assert not runner.is_alive()


def _loop_adapter(controller: ExecutionController) -> LoopAdapter:
    loop = controller._loop  # pyright: ignore[reportPrivateUsage]
    ralph = loop._ralph  # pyright: ignore[reportPrivateUsage]
    return cast(LoopAdapter, ralph)


def test_controller_session_reaches_worker_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    graph = _build_graph(repo)
    try:
        worker = _install_worker(tmp_path, monkeypatch)
        config = MilknadoConfig(
            execution_agent=worker,
            flavors={"implement": FlavorOverride(review=False)},
            quality_gates=(Gate(command="true"),),
            worktree_pattern="milknado-wt-{node_id}-{slug}",
            concurrency_limit=1,
            project_root=repo,
            db_path=repo / ".milknado" / "graph.db",
        )
        controller = build_execution_controller(graph, config, repo)
        assert (
            controller.session_input(
                "unknown-run", SessionInput(action="follow_up", text="ignored")
            )
            is False
        )
        run_result, snapshot, run_id = _run_controller_session(controller)
        assert run_result.completed_total == 1
        assert run_result.failed_total == 0
        assert snapshot.failed == 0
        assert len(snapshot.terminal_runs) == 1
        terminal = snapshot.terminal_runs[0]
        assert terminal.run_id == run_id
        assert terminal.status.value == "completed"
        assert any(
            event.kind == "user"
            and event.state == "delivered"
            and event.text == _FOLLOW_UP
            and event.event_id.endswith(_REQUEST_ID)
            for event in terminal.session.events
        )
        assert any(
            event.kind == "status" and event.event_id.endswith(_SESSION_ID)
            for event in terminal.session.events
        )
        ralph = _loop_adapter(controller)
        assert ralph.get_run_session_id(run_id) == _SESSION_ID
        assert ralph.get_run_session(run_id) == terminal.session
        assert (repo / "guidance.txt").read_text(encoding="utf-8") == _FOLLOW_UP
        assert (
            controller.session_input(run_id, SessionInput(action="follow_up", text="late input"))
            is False
        )
        assert (tmp_path / "received.txt").read_text(encoding="utf-8") == _FOLLOW_UP

        graph.close()
        replay_graph = MikadoGraph(repo / ".milknado" / "graph.db")
        try:
            replay = replay_graph.sessions.view(run_id)
            assert replay.active is False
            assert replay.context == terminal.session.context
            assert any(
                event.kind == "user"
                and event.state == "delivered"
                and event.text == _FOLLOW_UP
                and event.event_id.endswith(_REQUEST_ID)
                for event in replay.events
            )
            assert any(event.event_id.endswith(_SESSION_ID) for event in replay.events)
        finally:
            replay_graph.close()
    finally:
        graph.close()
