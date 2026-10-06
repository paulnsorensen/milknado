from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest

from milknado.adapters.coordinator_turns import NativeCoordinatorTurns
from milknado.domains.common import MilknadoConfig, SessionContext, SessionEvent, SessionInput
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator.control_models import (
    CoordinatorCommandReceipt,
    RuntimeAction,
    StartGoal,
    StartTurn,
)
from milknado.domains.coordinator.receipt_results import reserve_command_receipt
from milknado.domains.graph import MikadoGraph
from milknado.loop._agent import AgentResult
from milknado.loop.sessions import RuntimeRequest, RuntimeResult, RuntimeSession, SessionChannel


def _active_turn(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[MikadoGraph, CoordinatorControl, str, RuntimeSession, Event, Thread]:
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
    ready, finish = Event(), Event()

    def execute(request: RuntimeRequest) -> RuntimeResult:
        request.channel.start(SessionContext(family="codex", cwd=str(root)), ("interrupt",))
        assert request.spec.on_session_id is not None
        request.spec.on_session_id("thread")
        ready.set()
        assert finish.wait(2)
        request.channel.close()
        return RuntimeResult(AgentResult(130, session_id="thread", terminal_confirmed=False))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    control = CoordinatorControl(
        graph,
        root,
        CoordinatorServices(
            turn_runtime=adapter,
            turn_owner=adapter.owner,
            runtime_session=adapter.runtime_session,
        ),
    )
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    turn = Thread(
        target=lambda: control.send_coordinator_command(session_id, StartTurn("turn", "Work")),
        daemon=True,
    )
    turn.start()
    assert ready.wait(2)
    active = adapter.runtime_session("thread")
    assert active is not None
    return graph, control, session_id, active, finish, turn


def test_concurrent_output_and_intervention_do_not_invert_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, control, session_id, active, finish, turn = _active_turn(tmp_path, monkeypatch)
    reserved, release, publishing = Event(), Event(), Event()
    original_persist = SessionChannel._persist  # pyright: ignore[reportPrivateUsage]

    def reserve(
        conn: sqlite3.Connection, session_id: str, command_id: str, fingerprint: str
    ) -> CoordinatorCommandReceipt | None:
        if command_id == "action":
            reserved.set()
            assert release.wait(2)
        return reserve_command_receipt(conn, session_id, command_id, fingerprint)

    def persist(channel: SessionChannel, event: SessionEvent) -> None:
        if event.text == "Output":
            publishing.set()
        original_persist(channel, event)

    monkeypatch.setattr(
        "milknado.domains.coordinator.runtime_actions.reserve_command_receipt", reserve
    )
    monkeypatch.setattr(SessionChannel, "_persist", persist)
    actions: list[str] = []
    action = Thread(
        target=lambda: actions.append(
            control.send_coordinator_command(
                session_id, RuntimeAction("action", "thread", SessionInput(action="interrupt"))
            ).status
        ),
        daemon=True,
    )
    action.start()
    assert reserved.wait(2)
    output = Thread(
        target=lambda: active.channel.publish(SessionEvent(kind="assistant", text="Output")),
        daemon=True,
    )
    output.start()
    assert publishing.wait(2)
    release.set()
    action.join(timeout=2)
    output.join(timeout=2)
    finish.set()
    turn.join(timeout=2)
    assert not action.is_alive() and not output.is_alive() and not turn.is_alive()
    assert actions == ["accepted"]
    assert any(
        event.kind == "assistant" and event.text == "Output"
        for event in control.read_coordinator_snapshot(session_id, 0).events
    )
    graph.close()
