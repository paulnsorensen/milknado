from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Event, Thread

import pytest

from milknado.adapters._group_action_session import GroupActionSession
from milknado.adapters.coordinator_turns import NativeCoordinatorTurns
from milknado.adapters.recovery import ExistingWorktreeRecovery
from milknado.domains.common import (
    MilknadoConfig,
    NodeKind,
    NodeSpec,
    SessionAction,
    SessionContext,
    SessionEvent,
    SessionInput,
)
from milknado.domains.coordinator import ActionSession
from milknado.domains.coordinator.commands import CoordinatorAction, submit_coordinator_action
from milknado.domains.coordinator.control_services import TurnRuntimeHooks, TurnRuntimeRequest
from milknado.domains.coordinator.model import CoordinatorSession, ProviderBinding
from milknado.domains.coordinator.persistence import bind_provider_session
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import ExecutionGroup, GoalReviewRequest, MikadoGraph, TaskAttempt
from milknado.loop._agent import AgentResult
from milknado.loop.sessions import RuntimeRequest, RuntimeResult


def _group_setup(
    root: Path,
) -> tuple[MikadoGraph, NativeCoordinatorTurns, CoordinatorSession, TurnRuntimeRequest, int, int]:
    graph = MikadoGraph(root / "graph.db")
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    task = graph.add_node("task", goal.id)
    assert graph.claim_node(task.id, "run", now="2026-09-12T12:00:00+00:00")
    graph.runs.start("run", task.id, "run.log", "2026-09-12T12:00:00+00:00", None)
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
    request = TurnRuntimeRequest(
        "codex",
        "Work",
        ExecutionGroup("group", "graph", str(root), "branch", None),
        None,
        TurnRuntimeHooks("owner", lambda _identity: None, lambda _event: None),
        TaskAttempt("group", task.id, "run", "run"),
    )
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session = CoordinatorWorkflow(graph, conn).start_goal("Deliver", "codex")
        bind_provider_session(
            conn,
            session.id,
            ProviderBinding("execution_group", "group", "codex", "provider"),
        )
    return graph, adapter, session, request, goal.id, task.id


def _execute_group(
    request: RuntimeRequest, root: Path, ready: Event, release: Event
) -> RuntimeResult:
    request.channel.start(
        SessionContext(family="codex", cwd=str(root)),
        ("steer", "follow_up", "approve", "deny", "interrupt"),
        invocation_id="invocation",
    )
    request.channel.publish(
        SessionEvent(kind="permission", text="Approve", event_id="ask", state="requested")
    )
    assert request.spec.on_session_id is not None
    request.spec.on_session_id("provider")
    ready.set()
    assert release.wait(timeout=3)
    request.channel.close()
    return RuntimeResult(AgentResult(0, session_id="provider"))


def _restore(_recovery: ExistingWorktreeRecovery, _group: ExecutionGroup) -> bool:
    return True


@contextmanager
def _active_group(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[MikadoGraph, CoordinatorSession, int, int, GroupActionSession]]:
    graph, adapter, session, turn, goal_id, task_id = _group_setup(root)
    ready, release = Event(), Event()
    errors: list[BaseException] = []

    def execute(request: RuntimeRequest) -> RuntimeResult:
        return _execute_group(request, root, ready, release)

    def run_turn() -> None:
        try:
            _ = adapter.run(turn)
        except BaseException as error:
            errors.append(error)
            ready.set()

    monkeypatch.setattr(
        "milknado.adapters.coordinator_turns.ExistingWorktreeRecovery.restore", _restore
    )
    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    worker = Thread(target=run_turn)
    worker.start()
    try:
        assert ready.wait(timeout=3)
        assert not errors
        runtime = adapter.runtime_session("provider")
        assert isinstance(runtime, GroupActionSession)
        yield graph, session, goal_id, task_id, runtime
    finally:
        release.set()
        worker.join(timeout=3)
        graph.close()
        assert not worker.is_alive()
        assert not errors


def _submit(
    graph: MikadoGraph, session: CoordinatorSession, runtime: ActionSession, action: SessionInput
) -> str:
    command = CoordinatorAction(action.command_id or action.action, action)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        return submit_coordinator_action(conn, session, runtime, command).state


def _review(graph: MikadoGraph, goal_id: int, task_id: int) -> None:
    _ = graph.request_goal_review(
        GoalReviewRequest(
            goal_id=goal_id,
            goal_revision="sha256:goal-contract",
            evidence="New evidence changes the outcome.",
            proposed_change="Change the agreed goal.",
            affected_node_ids=(task_id,),
            reviewer="worker",
        )
    )


@pytest.mark.parametrize("action", ["approve", "steer", "follow_up"])
def test_pending_review_refuses_group_runtime_action_before_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: SessionAction
) -> None:
    with _active_group(tmp_path, monkeypatch) as (graph, session, goal_id, task_id, runtime):
        channel = runtime.channel
        _review(graph, goal_id, task_id)
        request_id = "1/invocation/ask" if action == "approve" else ""
        assert (
            _submit(graph, session, runtime, SessionInput(action=action, request_id=request_id))
            == "rejected"
        )
        assert graph.commands.command(action) is None
        assert all(item.action != action for item in channel.drain())
        assert (
            _submit(
                graph,
                session,
                runtime,
                SessionInput(action="deny", request_id="1/invocation/ask"),
            )
            == "queued"
        )
        assert _submit(graph, session, runtime, SessionInput(action="interrupt")) == "queued"


def test_group_runtime_action_is_refused_at_claim_after_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _active_group(tmp_path, monkeypatch) as (graph, session, goal_id, task_id, runtime):
        action = SessionInput(action="steer", text="Continue", command_id="steer-command")
        assert _submit(graph, session, runtime, action) == "queued"
        assert _submit(graph, session, runtime, action) == "queued"
        assert graph.commands.command("steer-command") is not None
        assert len(graph.commands.history("steer-command")) == 1
        _review(graph, goal_id, task_id)
        assert all(item.action != "steer" for item in runtime.channel.drain())
        receipt = graph.commands.receipt("steer-command")
        assert receipt is not None and receipt.status == "rejected"


def _assert_capability_change_rejects_interrupt(
    graph: MikadoGraph, session: CoordinatorSession, runtime: GroupActionSession, node_id: int
) -> None:
    _ = graph.commands.publish_capabilities("run", node_id, "next", "owner", ("interrupt",))
    try:
        assert _submit(graph, session, runtime, SessionInput(action="interrupt")) == "rejected"
    finally:
        _ = graph.commands.publish_capabilities(
            "run",
            node_id,
            "invocation",
            "owner",
            ("steer", "follow_up", "approve", "deny", "interrupt"),
            ("1/invocation/ask",),
        )


def test_group_runtime_action_uses_exact_permission_and_current_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _active_group(tmp_path, monkeypatch) as (graph, session, _goal_id, _task_id, runtime):
        stale = SessionInput(action="approve", request_id="ask", command_id="stale-permission")
        assert _submit(graph, session, runtime, stale) == "rejected"
        assert graph.commands.command("stale-permission") is None

        current = SessionInput(
            action="approve", request_id="1/invocation/ask", command_id="current"
        )
        assert _submit(graph, session, runtime, current) == "queued"
        stored = graph.commands.command("current")
        assert stored is not None
        assert stored.permission_id == "1/invocation/ask"
        assert stored.invocation_id == "invocation"
        (claimed,) = runtime.channel.drain()
        assert claimed.request_id == "ask"

        stale_owner = SessionInput(
            action="interrupt", owner_incarnation="other", command_id="stale-owner"
        )
        assert _submit(graph, session, runtime, stale_owner) == "rejected"
        stale_invocation = SessionInput(
            action="interrupt", invocation_id="other", command_id="stale-invocation"
        )
        assert _submit(graph, session, runtime, stale_invocation) == "rejected"
        assert graph.commands.command("stale-owner") is None
        assert graph.commands.command("stale-invocation") is None

        _assert_capability_change_rejects_interrupt(graph, session, runtime, stored.node_id)
