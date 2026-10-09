from __future__ import annotations

import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, cast

import pytest

from milknado.adapters.coordinator_turns import NativeCoordinatorTurns
from milknado.adapters.recovery import ExistingWorktreeRecovery
from milknado.domains.common import MilknadoConfig
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator.control_models import (
    AttemptCommand,
    CreateGroup,
    DispatchTask,
    RequestGoalReview,
    StartGoal,
    StartTurn,
)
from milknado.domains.graph import ExecutionGroup, MikadoGraph
from milknado.loop._agent import AgentResult, AgentRunSpec
from milknado.loop._process_gate import SpawnOptions
from milknado.loop.sessions import RuntimeRequest, RuntimeResult, SessionChannel


def _adapter(
    root: Path, graph: MikadoGraph, command: str = "codex exec"
) -> NativeCoordinatorTurns:
    return NativeCoordinatorTurns(
        root,
        MilknadoConfig(
            project_root=root,
            db_path=graph.db_path,
            agent_family="codex",
            execution_agent=command,
        ),
        graph,
    )


def _launched_group(
    graph: MikadoGraph, control: CoordinatorControl, root: Path
) -> tuple[str, int, str, str]:
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    goal_id = cast(int, cast(dict[str, object], started.result)["goal_id"])
    task = graph.add_node("Implement", goal_id)
    created = control.send_coordinator_command(
        session_id, CreateGroup("group", "main", (task.id,), str(root / "worktree"), "branch")
    )
    group_id = cast(str, cast(dict[str, object], created.result)["id"])
    dispatched = control.send_coordinator_command(
        session_id, DispatchTask("dispatch", group_id, task.id, "run")
    )
    attempt = cast(dict[str, object], cast(dict[str, object], dispatched.result)["attempt"])
    attempt_id = cast(str, attempt["attempt_id"])
    launched = control.send_coordinator_command(
        session_id, AttemptCommand("launch", group_id, task.id, "run", attempt_id)
    )
    assert launched.status == "accepted"
    return session_id, task.id, group_id, attempt_id


def _review_before_spawn(
    control: CoordinatorControl, session_id: str, task_id: int, root: Path
) -> Callable[[RuntimeRequest], RuntimeResult]:
    def execute(request: RuntimeRequest) -> RuntimeResult:
        review = control.send_coordinator_command(
            session_id,
            RequestGoalReview("review", "rev", "evidence", "change", "agent", (task_id,)),
        )
        assert review.status == "accepted"
        assert request.spec.spawn_worker is not None
        _ = request.spec.spawn_worker(
            SpawnOptions(
                (sys.executable, "-c", "pass"),
                root,
                None,
                False,
                subprocess.DEVNULL,
                subprocess.PIPE,
                subprocess.PIPE,
            )
        )
        raise AssertionError("reviewed task launched")

    return execute


def test_review_between_preparation_and_spawn_blocks_group_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    adapter = _adapter(tmp_path, graph)
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(turn_runtime=adapter, turn_owner=adapter.owner)
    )
    session_id, task_id, group_id, attempt_id = _launched_group(graph, control, tmp_path)

    def restore(_recovery: ExistingWorktreeRecovery, _group: ExecutionGroup) -> bool:
        return True

    monkeypatch.setattr(ExistingWorktreeRecovery, "restore", restore)
    spawns: list[SpawnOptions] = []

    def record_spawn(options: SpawnOptions, _context: object) -> NoReturn:
        spawns.append(options)
        raise AssertionError("reviewed task spawned")

    execute = _review_before_spawn(control, session_id, task_id, tmp_path)
    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    monkeypatch.setattr("milknado.adapters.coordinator_turns.spawn_protected", record_spawn)
    receipt = control.send_coordinator_command(
        session_id,
        StartTurn(
            "turn",
            "Work",
            group_id=group_id,
            node_id=task_id,
            run_id="run",
            attempt_id=attempt_id,
        ),
    )
    assert receipt.status == "unavailable"
    assert spawns == []
    graph.close()


def test_invalid_resume_releases_fence_without_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    adapter = _adapter(tmp_path, graph, "codex resume old")
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(turn_runtime=adapter, turn_owner=adapter.owner)
    )
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    calls: list[list[str]] = []

    def execute(spec: AgentRunSpec, _channel: SessionChannel) -> AgentResult:
        calls.append(spec.cmd)
        return AgentResult(0, session_id="thread", terminal_confirmed=True)

    monkeypatch.setattr("milknado.loop.sessions._lifecycle.run_session", execute)
    first = control.send_coordinator_command(session_id, StartTurn("first", "Work"))
    assert first.status == "accepted"
    second = control.send_coordinator_command(session_id, StartTurn("second", "Work"))
    assert second.status == "unavailable"
    assert "resume command already selects a Codex thread" in cast(str, second.result)
    assert calls == [["codex", "resume", "old"]]
    adapter.shutdown(timeout=0.01)
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute(
            "SELECT state FROM coordinator_turn_launches ORDER BY rowid"
        ).fetchall() == [("confirmed",), ("unknown",)]
    graph.close()


def test_postlaunch_value_error_keeps_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    adapter = _adapter(tmp_path, graph)
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(turn_runtime=adapter, turn_owner=adapter.owner)
    )
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])

    def execute(_spec: AgentRunSpec, _channel: SessionChannel) -> NoReturn:
        raise ValueError("provider failed after worker launch")

    monkeypatch.setattr("milknado.loop.sessions._lifecycle.run_session", execute)
    receipt = control.send_coordinator_command(session_id, StartTurn("failed", "Work"))
    assert receipt.status == "unavailable"
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute(
            "SELECT state FROM coordinator_turn_launches WHERE command_id = 'failed'"
        ).fetchone() == ("submitted",)
    graph.close()
