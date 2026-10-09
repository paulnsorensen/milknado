from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from milknado.adapters.coordinator_turns import NativeCoordinatorTurns
from milknado.domains.common import MilknadoConfig, SessionContext, SessionEvent, SessionInput
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator.control_models import RuntimeAction, StartGoal, StartTurn
from milknado.domains.graph import MikadoGraph
from milknado.loop._agent import AgentResult
from milknado.loop.sessions import RuntimeRequest, RuntimeResult


def _bound_control(root: Path) -> tuple[MikadoGraph, CoordinatorControl, str]:
    graph = MikadoGraph(root / "graph.db")
    adapter = NativeCoordinatorTurns(
        root,
        MilknadoConfig(
            project_root=root,
            db_path=graph.db_path,
            agent_family="codex",
            execution_agent="codex exec",
        ),
    )
    control = CoordinatorControl(
        graph,
        root,
        CoordinatorServices(
            turn_runtime=adapter,
            turn_owner=adapter.owner,
            runtime_session=adapter.runtime_session,
        ),
    )
    goal = control.send_coordinator_command("", StartGoal("goal", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], goal.result)["id"])
    return graph, control, session_id


def _assert_stale_and_current_approval(
    control: CoordinatorControl,
    session_id: str,
    request: RuntimeRequest,
    permission_ids: list[str],
) -> None:
    stale = RuntimeAction(
        "stale", "thread", SessionInput(action="approve", request_id=permission_ids[0])
    )
    rejected = control.send_coordinator_command(session_id, stale)
    assert rejected.status == "accepted"
    assert cast(dict[str, object], rejected.result)["state"] == "rejected"
    assert control.send_coordinator_command(session_id, stale) == rejected

    current = RuntimeAction(
        "current", "thread", SessionInput(action="approve", request_id=permission_ids[-1])
    )
    accepted = control.send_coordinator_command(session_id, current)
    assert accepted.status == "accepted"
    assert cast(dict[str, object], accepted.result)["state"] == "queued"
    (submitted,) = request.channel.drain()
    assert submitted.request_id == "ask"


def _native_invoke(
    control: CoordinatorControl, session_id: str, root: Path, permission_ids: list[str]
) -> Callable[[RuntimeRequest], RuntimeResult]:
    def execute(request: RuntimeRequest) -> RuntimeResult:
        invocation_id = f"invocation-{len(permission_ids) + 1}"
        request.channel.start(
            SessionContext(family="codex", cwd=str(root)),
            ("approve",),
            invocation_id=invocation_id,
        )
        assert request.spec.on_session_id is not None
        request.spec.on_session_id("thread")
        request.channel.publish(
            SessionEvent(kind="permission", text="Approve tool", event_id="ask", state="requested")
        )
        snapshot = control.read_coordinator_snapshot(session_id, 0)
        current_id = next(
            event.entity_id
            for event in snapshot.events
            if event.kind == "permission" and event.turn_id == f"turn-{len(permission_ids) + 1}"
        )
        permission_ids.append(current_id)
        if len(permission_ids) == 2:
            _assert_stale_and_current_approval(control, session_id, request, permission_ids)
        request.channel.close()
        return RuntimeResult(AgentResult(0, session_id="thread", terminal_confirmed=False))

    return execute


def test_resumed_native_turn_rejects_delayed_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, control, session_id = _bound_control(tmp_path)
    permission_ids: list[str] = []
    execute = _native_invoke(control, session_id, tmp_path, permission_ids)
    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    assert (
        control.send_coordinator_command(session_id, StartTurn("turn-1", "Work")).status
        == "unavailable"
    )
    assert (
        control.send_coordinator_command(session_id, StartTurn("turn-2", "Continue")).status
        == "unavailable"
    )
    assert len(set(permission_ids)) == 2
    graph.close()
