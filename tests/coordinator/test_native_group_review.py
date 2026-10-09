from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread
from typing import NoReturn, cast

import pytest

from milknado.adapters.coordinator_turns import NativeCoordinatorTurns
from milknado.domains.common import MilknadoConfig, SessionContext, SessionInput
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator.control_models import (
    AttemptCommand,
    CreateGroup,
    DispatchTask,
    FinishTask,
    RequestGoalReview,
    StartGoal,
    StartTurn,
)
from milknado.domains.graph import ExecutionGroup, MikadoGraph
from milknado.loop._agent import AgentResult
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions
from milknado.loop.sessions import RuntimeRequest, RuntimeResult


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
    dispatch = control.send_coordinator_command(
        session_id, DispatchTask("dispatch", group_id, task.id, "run")
    )
    attempt = cast(dict[str, object], cast(dict[str, object], dispatch.result)["attempt"])
    attempt_id = cast(str, attempt["attempt_id"])
    launched = control.send_coordinator_command(
        session_id, AttemptCommand("launch", group_id, task.id, "run", attempt_id)
    )
    assert launched.status == "accepted"
    return session_id, task.id, group_id, attempt_id


def _restore(_recovery: object, _group: ExecutionGroup) -> bool:
    return True


def test_review_after_group_launch_interrupts_native_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    adapter = NativeCoordinatorTurns(
        tmp_path,
        MilknadoConfig(
            project_root=tmp_path,
            db_path=graph.db_path,
            agent_family="codex",
            execution_agent="codex exec",
        ),
        graph,
    )
    monkeypatch.setattr(
        "milknado.adapters.coordinator_turns.ExistingWorktreeRecovery.restore",
        _restore,
    )
    ready, reviewed = Event(), Event()
    received: list[SessionInput] = []

    def execute(request: RuntimeRequest) -> RuntimeResult:
        request.channel.start(
            SessionContext(family="codex", cwd=str(request.spec.cwd)),
            ("interrupt",),
            invocation_id="invocation",
        )
        assert request.spec.on_session_id is not None
        request.spec.on_session_id("provider")
        ready.set()
        assert reviewed.wait(2)
        received.extend(request.channel.drain())
        request.channel.close()
        return RuntimeResult(AgentResult(130, session_id="provider", terminal_confirmed=False))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    control = CoordinatorControl(
        graph,
        tmp_path,
        CoordinatorServices(turn_runtime=adapter, turn_owner=adapter.owner),
    )
    session_id, task_id, group_id, attempt_id = _launched_group(graph, control, tmp_path)
    result: list[str] = []
    turn = Thread(
        target=lambda: result.append(
            control.send_coordinator_command(
                session_id,
                StartTurn(
                    "turn",
                    "Work",
                    group_id=group_id,
                    node_id=task_id,
                    run_id="run",
                    attempt_id=attempt_id,
                ),
            ).status
        ),
        daemon=True,
    )
    turn.start()
    assert ready.wait(2)
    capabilities = graph.commands.capabilities(attempt_id)
    assert capabilities is not None
    assert capabilities.owner_incarnation == "turn"
    review = control.send_coordinator_command(
        session_id,
        RequestGoalReview("review", "rev", "evidence", "change", "agent", (task_id,)),
    )
    assert review.status == "accepted"
    queued = graph.commands.pending(attempt_id)
    assert len(queued) == 1 and queued[0].action == "interrupt"
    reviewed.set()
    turn.join(timeout=2)
    assert not turn.is_alive()
    assert len(received) == 1
    assert received[0].command_id == queued[0].command_id
    assert received[0].action == "interrupt"
    assert result == ["unavailable"]
    graph.close()


@pytest.mark.skipif(os.name == "nt", reason="protected worker requires POSIX")
def test_finish_task_cannot_clear_live_native_group_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    adapter = NativeCoordinatorTurns(
        tmp_path,
        MilknadoConfig(
            project_root=tmp_path,
            db_path=graph.db_path,
            agent_family="codex",
            execution_agent="codex exec",
        ),
        graph,
    )
    monkeypatch.setattr(
        "milknado.adapters.coordinator_turns.ExistingWorktreeRecovery.restore", _restore
    )
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(turn_runtime=adapter, turn_owner=adapter.owner)
    )
    session_id, task_id, group_id, attempt_id = _launched_group(graph, control, tmp_path)

    def execute(request: RuntimeRequest) -> RuntimeResult:
        assert request.spec.spawn_worker is not None
        worker = request.spec.spawn_worker(
            SpawnOptions(
                (sys.executable, "-c", "import time; time.sleep(30)"),
                tmp_path,
                None,
                False,
                subprocess.DEVNULL,
                subprocess.PIPE,
                subprocess.PIPE,
            )
        )
        try:
            with sqlite3.connect(graph.db_path) as conn:
                assert conn.execute(
                    "SELECT graph_run_id, node_id FROM run_workers WHERE runtime_run_id = 'turn'"
                ).fetchone() == (attempt_id, task_id)
            finish = control.send_coordinator_command(
                session_id,
                FinishTask("finish-live", group_id, task_id, "run", attempt_id, True, "done"),
            )
            assert finish.status == "rejected"
            assert graph.groups.active_attempt(group_id) is not None
        finally:
            assert worker.cleanup()
        return RuntimeResult(AgentResult(0, session_id="provider", terminal_confirmed=False))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    result = control.send_coordinator_command(
        session_id,
        StartTurn(
            "turn", "Work", group_id=group_id, node_id=task_id, run_id="run", attempt_id=attempt_id
        ),
    )
    assert result.status == "unavailable"
    graph.close()


def _two_task_group(
    graph: MikadoGraph, control: CoordinatorControl, root: Path
) -> tuple[str, int, int, str, str]:
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    goal_id = cast(int, cast(dict[str, object], started.result)["goal_id"])
    first = graph.add_node("First", goal_id)
    second = graph.add_node("Second", goal_id)
    created = control.send_coordinator_command(
        session_id,
        CreateGroup("group", "main", (first.id, second.id), str(root / "worktree"), "branch"),
    )
    group_id = cast(str, cast(dict[str, object], created.result)["id"])
    dispatched = control.send_coordinator_command(
        session_id, DispatchTask("dispatch-a", group_id, first.id, "run-a")
    )
    attempt = cast(dict[str, object], cast(dict[str, object], dispatched.result)["attempt"])
    attempt_id = cast(str, attempt["attempt_id"])
    assert (
        control.send_coordinator_command(
            session_id, AttemptCommand("launch-a", group_id, first.id, "run-a", attempt_id)
        ).status
        == "accepted"
    )
    return session_id, first.id, second.id, group_id, attempt_id


def test_stale_group_turn_cannot_spawn_on_successor_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    adapter = NativeCoordinatorTurns(
        tmp_path,
        MilknadoConfig(
            project_root=tmp_path,
            db_path=graph.db_path,
            agent_family="codex",
            execution_agent="codex exec",
        ),
        graph,
    )
    monkeypatch.setattr(
        "milknado.adapters.coordinator_turns.ExistingWorktreeRecovery.restore", _restore
    )
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(turn_runtime=adapter, turn_owner=adapter.owner)
    )
    session_id, first_id, second_id, group_id, attempt_a = _two_task_group(
        graph, control, tmp_path
    )
    paused, resume = Event(), Event()
    spawns: list[str | None] = []

    def execute(request: RuntimeRequest) -> RuntimeResult:
        assert request.spec.spawn_worker is not None
        paused.set()
        assert resume.wait(2)
        _ = request.spec.spawn_worker(
            SpawnOptions(
                (sys.executable, "-c", "pass"),
                tmp_path,
                None,
                False,
                subprocess.DEVNULL,
                subprocess.PIPE,
                subprocess.PIPE,
            )
        )
        return RuntimeResult(AgentResult(0))

    def record_spawn(_options: SpawnOptions, context: ProtectionContext) -> NoReturn:
        spawns.append(context.owner.graph_run_id)
        raise ValueError("unexpected spawn")

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    monkeypatch.setattr("milknado.adapters.coordinator_turns.spawn_protected", record_spawn)
    results: list[str] = []

    def send_turn() -> None:
        receipt = control.send_coordinator_command(
            session_id,
            StartTurn(
                "turn-a",
                "Work",
                group_id=group_id,
                node_id=first_id,
                run_id="run-a",
                attempt_id=attempt_a,
            ),
        )
        results.append(receipt.status)

    turn = Thread(target=send_turn, daemon=True)
    turn.start()
    assert paused.wait(2)
    assert (
        control.send_coordinator_command(
            session_id,
            FinishTask("finish-a", group_id, first_id, "run-a", attempt_a, True, "done"),
        ).status
        == "accepted"
    )
    dispatched_b = control.send_coordinator_command(
        session_id, DispatchTask("dispatch-b", group_id, second_id, "run-b")
    )
    next_attempt = cast(dict[str, object], cast(dict[str, object], dispatched_b.result)["attempt"])
    attempt_b = cast(str, next_attempt["attempt_id"])
    assert (
        control.send_coordinator_command(
            session_id, AttemptCommand("launch-b", group_id, second_id, "run-b", attempt_b)
        ).status
        == "accepted"
    )
    resume.set()
    turn.join(timeout=2)
    assert not turn.is_alive()
    assert results == ["unavailable"]
    assert spawns == []
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM run_workers WHERE graph_run_id = ?", (attempt_b,)
        ).fetchone() == (0,)
    graph.close()
