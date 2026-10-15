"""Native retry keeps the real turn, approval, and cancel gates."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest

from milknado.adapters.coordinator_turns import NativeCoordinatorTurns
from milknado.adapters.recovery import ExistingWorktreeRecovery
from milknado.domains.common import (
    MikadoNode,
    MilknadoConfig,
    SessionContext,
    SessionEvent,
    SessionInput,
)
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator.control_models import CancelTurn, RuntimeAction, StartTurn
from milknado.domains.coordinator.model import CoordinatorSession
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import ExecutionGroup, GroupWorkspace, MikadoGraph, TaskAttempt
from milknado.loop._agent import AgentResult
from milknado.loop.sessions import RuntimeRequest, RuntimeResult


def _turn(turn_id: str, attempt: TaskAttempt) -> StartTurn:
    return StartTurn(
        turn_id,
        "Work",
        group_id=attempt.group_id,
        node_id=attempt.node_id,
        run_id=attempt.run_id,
        attempt_id=attempt.attempt_id,
    )


def _permission_id(control: CoordinatorControl, session_id: str, turn_id: str) -> str:
    snapshot = control.read_coordinator_snapshot(session_id, 0)
    return next(
        event.entity_id
        for event in snapshot.events
        if event.kind == "permission" and event.turn_id == turn_id
    )


def _assert_retry_approval(
    control: CoordinatorControl, session_id: str, permission_ids: list[str]
) -> None:
    stale = control.send_coordinator_command(
        session_id,
        RuntimeAction(
            "stale-approval",
            "provider",
            SessionInput(action="approve", request_id=permission_ids[0]),
        ),
    )
    assert stale.status == "accepted"
    assert cast(dict[str, object], stale.result)["state"] == "rejected"
    current = control.send_coordinator_command(
        session_id,
        RuntimeAction(
            "current-approval",
            "provider",
            SessionInput(action="approve", request_id=permission_ids[1]),
        ),
    )
    assert current.status == "accepted"
    assert cast(dict[str, object], current.result)["state"] == "queued"


def _start_retry_turn(
    control: CoordinatorControl, session_id: str, attempt: TaskAttempt, ready: Event
) -> tuple[Thread, list[str]]:
    results: list[str] = []
    thread = Thread(
        target=lambda: results.append(
            control.send_coordinator_command(session_id, _turn("turn-2", attempt)).status
        )
    )
    thread.start()
    assert ready.wait(timeout=3), results
    return thread, results


def _cancel_retry_turn(control: CoordinatorControl, session_id: str, stopped: Event) -> None:
    cancel = control.send_coordinator_command(session_id, CancelTurn("cancel", "turn-2"))
    assert cancel.status == "accepted"
    assert stopped.wait(timeout=3)


def _restore(_recovery: ExistingWorktreeRecovery, _group: ExecutionGroup) -> bool:
    return True


def _native_control(
    root: Path, graph: MikadoGraph
) -> tuple[NativeCoordinatorTurns, CoordinatorControl]:
    adapter = NativeCoordinatorTurns(
        root,
        MilknadoConfig(
            project_root=root,
            db_path=graph.db_path,
            agent_family="codex",
            execution_agent="codex exec",
        ),
        graph,
    )
    control = CoordinatorControl(
        graph,
        root,
        CoordinatorServices(
            turn_runtime=adapter,
            turn_owner=adapter.owner,
            turn_cancel=adapter.cancel,
            runtime_session=adapter.runtime_session,
        ),
    )
    return adapter, control


def _retry_setup(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    MikadoGraph,
    CoordinatorWorkflow,
    CoordinatorSession,
    MikadoNode,
    ExecutionGroup,
    NativeCoordinatorTurns,
    CoordinatorControl,
]:
    _ = (root / "milknado.toml").write_text("[milknado]\nmax_node_attempts = 2\n")
    graph = MikadoGraph(root / "graph.db")
    workflow = CoordinatorWorkflow(graph, graph.group_connection, max_node_attempts=2)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    group = workflow.create_group(
        session,
        "main",
        (task.id,),
        GroupWorkspace(str(root / "group"), "branch", None),
    )
    adapter, control = _native_control(root, graph)
    monkeypatch.setattr(
        "milknado.adapters.coordinator_turns.ExistingWorktreeRecovery.restore",
        _restore,
    )
    return graph, workflow, session, task, group, adapter, control


def _runtime_double(
    root: Path,
    controlled: tuple[CoordinatorControl, str],
    gates: tuple[Event, Event],
    permission_ids: list[str],
) -> Callable[[RuntimeRequest], RuntimeResult]:
    control, session_id = controlled
    ready, stopped = gates

    def execute(request: RuntimeRequest) -> RuntimeResult:
        invocation = f"invocation-{len(permission_ids) + 1}"
        request.channel.start(
            SessionContext(family="codex", cwd=str(root)),
            ("approve", "interrupt"),
            invocation_id=invocation,
        )
        assert request.spec.on_session_id is not None
        request.spec.on_session_id("provider")
        request.channel.publish(
            SessionEvent(kind="permission", text="Approve tool", event_id="ask", state="requested")
        )
        turn_id = f"turn-{len(permission_ids) + 1}"
        permission_ids.append(_permission_id(control, session_id, turn_id))
        if len(permission_ids) == 2:
            ready.set()
            stop = request.spec.force_stop_event
            assert stop is not None and stop.wait(timeout=3)
            stopped.set()
        request.channel.close()
        return RuntimeResult(
            AgentResult(0, session_id="provider", terminal_confirmed=len(permission_ids) == 1)
        )

    return execute


def _fail_first_attempt(
    graph: MikadoGraph,
    workflow: CoordinatorWorkflow,
    control: CoordinatorControl,
    placed: tuple[CoordinatorSession, ExecutionGroup, MikadoNode],
) -> None:
    session, group, task = placed
    first = workflow.dispatch_task(session, group.id, task.id, "run-1")
    _ = workflow.acknowledge_launch(session, first)
    assert (
        control.send_coordinator_command(session.id, _turn("turn-1", first.attempt)).status
        == "accepted"
    )
    assert graph.task_attempts.confirm_execution(
        first.attempt.attempt_id, first.attempt.attempt_id, first.attempt.attempt_id
    )
    workflow.finish_task(session, first.attempt, NodeLoopOutcome(task.id, False, "failed"))
    assert graph.groups.task_result(task.id) is None


def test_native_retry_rejects_old_approval_and_cancels_current_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, workflow, session, task, group, adapter, control = _retry_setup(tmp_path, monkeypatch)
    ready = Event()
    stopped = Event()
    permission_ids: list[str] = []
    execute = _runtime_double(tmp_path, (control, session.id), (ready, stopped), permission_ids)
    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    _fail_first_attempt(graph, workflow, control, (session, group, task))

    second = workflow.dispatch_task(session, group.id, task.id, "run-2")
    _ = workflow.acknowledge_launch(session, second)
    thread, results = _start_retry_turn(control, session.id, second.attempt, ready)
    try:
        _assert_retry_approval(control, session.id, permission_ids)
        _cancel_retry_turn(control, session.id, stopped)
    finally:
        _ = adapter.cancel("turn-2")
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert results == ["unavailable"]
    assert graph.groups.active_attempt(group.id) == second.attempt
    assert graph.task_attempts.failed_count(task.id) == 1
    graph.close()
