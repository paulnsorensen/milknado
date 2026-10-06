from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest

from milknado.adapters.coordinator_turns import NativeCoordinatorTurns
from milknado.domains.common import MilknadoConfig, SessionContext, SessionEvent, SessionInput
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator.commands import CoordinatorAction, submit_coordinator_action
from milknado.domains.coordinator.control_models import (
    CancelTurn,
    CoordinatorCommandReceipt,
    RuntimeAction,
    StartGoal,
    StartTurn,
)
from milknado.domains.coordinator.control_services import TurnRuntimeHooks, TurnRuntimeRequest
from milknado.domains.coordinator.model import ProviderBinding
from milknado.domains.coordinator.persistence import bind_provider_session, link_entity
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import MikadoGraph
from milknado.loop._agent import AgentResult
from milknado.loop._process_gate import SpawnOptions
from milknado.loop.sessions import (
    ProviderSessionIdentity,
    RuntimeRequest,
    RuntimeResult,
    RuntimeSession,
    SessionChannel,
    submit_runtime_action,
)


def test_native_turn_registers_active_channel_and_protected_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = NativeCoordinatorTurns(
        tmp_path,
        MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / "graph.db",
            agent_family="codex",
            execution_agent="codex exec",
        ),
    )
    events: list[SessionEvent] = []
    identities: list[str] = []

    def execute(request: RuntimeRequest) -> RuntimeResult:
        assert request.spec.spawn_worker is not None
        assert request.spec.on_session_id is not None
        request.channel.start(
            SessionContext(family="codex", cwd=str(tmp_path)),
            ("steer", "follow_up", "interrupt", "approve", "deny"),
        )
        request.spec.on_session_id("thread")
        active = adapter.runtime_session("thread")
        assert active is not None
        assert (
            submit_runtime_action("thread", SessionInput(action="interrupt"), active).state
            == "queued"
        )
        request.channel.publish(
            SessionEvent(kind="permission", text="Approve tool", event_id="ask", state="requested")
        )
        assert (
            submit_runtime_action(
                "thread", SessionInput(action="approve", request_id="1/ask"), active
            ).state
            == "queued"
        )
        request.channel.publish(SessionEvent(kind="assistant", text="Streamed answer"))
        request.channel.close()
        return RuntimeResult(AgentResult(0, session_id="thread", terminal_confirmed=True))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    result = adapter.run(
        TurnRuntimeRequest(
            "codex",
            "Work",
            None,
            None,
            TurnRuntimeHooks("turn", identities.append, events.append),
        )
    )
    assert result.run is not None and result.run.session_id == "thread"
    assert identities == ["thread"]
    assert any(event.kind == "assistant" and event.text == "Streamed answer" for event in events)
    assert adapter.runtime_session("thread") is None


def test_group_provider_action_uses_group_binding_not_coordinator_family(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session = CoordinatorWorkflow(graph, conn).start_goal("Deliver", "codex")
        bind_provider_session(
            conn, session.id, ProviderBinding("execution_group", "group", "claude", "session")
        )
        link_entity(conn, session.id, "provider_session", "session")
        channel = SessionChannel()
        channel.start(SessionContext(family="claude", cwd=str(tmp_path)), ("interrupt",))
        incarnation = channel.capture_incarnation()
        assert incarnation is not None
        runtime = RuntimeSession(
            ProviderSessionIdentity("claude", "session"), channel, incarnation
        )
        receipt = submit_coordinator_action(
            conn,
            session,
            runtime,
            CoordinatorAction("interrupt", SessionInput(action="interrupt")),
        )
        assert receipt.state == "queued"
        assert channel.drain()[0].action == "interrupt"
    graph.close()


def test_first_identity_and_stream_are_durable_at_native_handshake(
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
    )
    observed: list[bool] = []

    def execute(request: RuntimeRequest) -> RuntimeResult:
        assert request.spec.on_session_id is not None
        request.channel.start(
            SessionContext(family="codex", cwd=str(tmp_path)), ("interrupt", "approve")
        )
        request.spec.on_session_id("thread")
        snapshot = control.read_coordinator_snapshot(session_id, 0)
        observed.append(
            snapshot.provider_bindings[0].provider_session_id == "thread"
            and snapshot.provider_turns[0].status == "submitted"
        )
        request.channel.publish(SessionEvent(kind="assistant", text="Before terminal"))
        request.channel.publish(
            SessionEvent(kind="permission", text="Approve tool", event_id="ask", state="requested")
        )
        snapshot = control.read_coordinator_snapshot(session_id, 0)
        permission = next(event for event in snapshot.events if event.kind == "permission")
        assert permission.entity_kind == "permission"
        assert permission.entity_id == "1/ask"
        assert (permission.turn_id, permission.provider_session_id) == ("turn", "thread")
        approval = control.send_coordinator_command(
            session_id,
            RuntimeAction(
                "approve",
                "thread",
                SessionInput(action="approve", request_id=permission.entity_id),
            ),
        )
        assert approval.status == "accepted"
        assert cast(dict[str, object], approval.result)["state"] == "queued"
        request.channel.close()
        return RuntimeResult(AgentResult(0, session_id="thread", terminal_confirmed=False))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    control = CoordinatorControl(
        graph,
        tmp_path,
        CoordinatorServices(
            turn_runtime=adapter,
            turn_owner=adapter.owner,
            runtime_session=adapter.runtime_session,
        ),
    )
    goal = control.send_coordinator_command("", StartGoal("goal", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], goal.result)["id"])
    assert (
        control.send_coordinator_command(session_id, StartTurn("turn", "Work")).status
        == "unavailable"
    )
    assert observed == [True]
    assert any(
        item.kind == "assistant" and item.text == "Before terminal"
        for item in control.read_coordinator_snapshot(session_id, 0).events
    )
    graph.close()


def test_cancel_turn_stops_waiting_native_run_without_completion(
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
    )
    waiting = Event()

    def execute(request: RuntimeRequest) -> RuntimeResult:
        stop = request.spec.force_stop_event
        assert stop is not None and request.spec.on_session_id is not None
        request.channel.start(SessionContext(family="codex", cwd=str(tmp_path)), ("interrupt",))
        request.spec.on_session_id("thread")
        waiting.set()
        assert stop.wait(timeout=2), "cancel did not stop the waiting native turn"
        request.channel.close()
        return RuntimeResult(AgentResult(130, session_id="thread", terminal_confirmed=False))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    control = CoordinatorControl(
        graph,
        tmp_path,
        CoordinatorServices(
            turn_runtime=adapter,
            turn_owner=adapter.owner,
            turn_cancel=adapter.cancel,
        ),
    )
    goal = control.send_coordinator_command("", StartGoal("goal", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], goal.result)["id"])
    results: list[CoordinatorCommandReceipt] = []
    worker = Thread(
        target=lambda: results.append(
            control.send_coordinator_command(session_id, StartTurn("turn", "Work"))
        )
    )
    worker.start()
    assert waiting.wait(timeout=2)
    cancel = control.send_coordinator_command(session_id, CancelTurn("cancel", "turn"))
    assert cancel.status == "accepted"
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(results) == 1 and results[0].status == "unavailable"
    assert [
        (turn.turn_id, turn.status)
        for turn in control.read_coordinator_snapshot(session_id, 0).provider_turns
    ] == [("turn", "submitted")]
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute(
            "SELECT state FROM coordinator_turn_launches WHERE command_id = 'turn'"
        ).fetchone() == ("unknown",)
    graph.close()


@pytest.mark.skipif(os.name == "nt", reason="protected worker requires POSIX")
def test_native_turn_spawn_owns_and_reaps_worker(
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
    )

    def execute(request: RuntimeRequest) -> RuntimeResult:
        spawn = request.spec.spawn_worker
        assert spawn is not None
        worker = spawn(
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
                    "SELECT runtime_run_id, graph_run_id FROM run_workers "
                    + "WHERE runtime_run_id = 'turn'"
                ).fetchone() == ("turn", None)
        finally:
            assert worker.cleanup()
        assert worker.process.poll() is not None
        return RuntimeResult(AgentResult(0))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    _ = adapter.run(
        TurnRuntimeRequest(
            "codex",
            "Work",
            None,
            None,
            TurnRuntimeHooks("turn", lambda _identity: None, lambda _event: None),
        )
    )
    graph.close()
