from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import SessionAction, SessionContext, SessionInput
from milknado.domains.coordinator import ControlEvent, CoordinatorSession, ProviderBinding
from milknado.domains.coordinator.commands import CoordinatorAction, submit_coordinator_action
from milknado.domains.coordinator.journal import append_control_event, control_history
from milknado.domains.coordinator.persistence import bind_provider_session
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import MikadoGraph
from milknado.loop.sessions import (
    ProviderSessionIdentity,
    RuntimeActionReceipt,
    RuntimeSession,
    SessionChannel,
    submit_runtime_action,
)


def _runtime(
    tmp_path: Path, actions: tuple[SessionAction, ...]
) -> tuple[MikadoGraph, sqlite3.Connection, CoordinatorSession, SessionChannel, RuntimeSession]:
    graph = MikadoGraph(tmp_path / "graph.db")
    conn = sqlite3.connect(graph.db_path)
    session = CoordinatorWorkflow(graph, conn).start_goal("Goal", "codex")
    bind_provider_session(
        conn, session.id, ProviderBinding("coordinator", session.id, "codex", "provider")
    )
    channel = SessionChannel()
    channel.start(SessionContext(family="codex", cwd=str(tmp_path)), actions)
    incarnation = channel.capture_incarnation()
    assert incarnation is not None
    runtime = RuntimeSession(ProviderSessionIdentity("codex", "provider"), channel, incarnation)
    return graph, conn, session, channel, runtime


def test_runtime_submission_keeps_admission_states_and_receipt_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, conn, session, channel, runtime = _runtime(tmp_path, ("steer", "approve"))

    def inspect_receipt(
        provider_session_id: str, action: SessionInput, current: RuntimeSession
    ) -> RuntimeActionReceipt:
        row = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT state FROM coordinator_action_receipts WHERE command_id = ?",
                (action.text,),
            ).fetchone(),
        )
        assert row == ("unconfirmed",)
        return submit_runtime_action(provider_session_id, action, current)

    monkeypatch.setattr("milknado.loop.sessions._lifecycle.submit_runtime_action", inspect_receipt)
    states = [
        submit_coordinator_action(conn, session, runtime, CoordinatorAction(name, action)).state
        for name, action in (
            ("queued", SessionInput(action="steer", text="queued")),
            ("rejected", SessionInput(action="approve", text="rejected")),
            ("unsupported", SessionInput(action="follow_up", text="unsupported")),
        )
    ]
    channel.close()
    stale = submit_coordinator_action(
        conn,
        session,
        runtime,
        CoordinatorAction("unknown_session", SessionInput(action="steer", text="unknown_session")),
    )
    assert states == ["queued", "rejected", "unsupported"]
    assert stale.state == "unknown_session"
    conn.close()
    graph.close()


def test_submit_failure_keeps_unconfirmed_receipt_and_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, conn, session, channel, runtime = _runtime(tmp_path, ("steer",))
    command = CoordinatorAction("failed", SessionInput(action="steer", text="continue"))
    calls = 0

    def fail_submit(
        _provider_session_id: str, _action: SessionInput, _runtime: RuntimeSession
    ) -> RuntimeActionReceipt:
        nonlocal calls
        calls += 1
        raise RuntimeError("injected submission failure")

    monkeypatch.setattr("milknado.loop.sessions._lifecycle.submit_runtime_action", fail_submit)
    with pytest.raises(RuntimeError, match="injected submission"):
        _ = submit_coordinator_action(conn, session, runtime, command)
    receipt = submit_coordinator_action(conn, session, runtime, command)
    assert receipt.state == "unconfirmed"
    assert calls == 1
    assert conn.execute(
        "SELECT state FROM coordinator_action_receipts WHERE command_id = 'failed'"
    ).fetchone() == ("unconfirmed",)
    assert channel.drain() == ()
    conn.close()
    graph.close()


def test_action_retry_repairs_history_without_resubmission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, conn, session, channel, runtime = _runtime(tmp_path, ("steer",))
    command = CoordinatorAction("command-1", SessionInput(action="steer", text="continue"))
    original = append_control_event
    failed = False

    def fail_once(target: sqlite3.Connection, session_id: str, event: ControlEvent) -> int:
        nonlocal failed
        if not failed:
            failed = True
            raise sqlite3.OperationalError("injected event failure")
        return original(target, session_id, event)

    monkeypatch.setattr("milknado.domains.coordinator.commands.append_control_event", fail_once)
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        _ = submit_coordinator_action(conn, session, runtime, command)
    receipt = submit_coordinator_action(conn, session, runtime, command)
    assert receipt.state == "queued"
    assert len(channel.drain()) == 1
    assert [event.kind for event in control_history(conn, session.id)] == ["command"]
    conn.close()
    graph.close()
