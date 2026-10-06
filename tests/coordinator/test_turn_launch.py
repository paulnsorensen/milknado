from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from threading import Thread
from typing import cast

import pytest
from typing_extensions import override

from milknado.adapters.coordinator_turns import NativeCoordinatorTurns
from milknado.domains.common import MilknadoConfig, SessionEvent
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator.control_models import (
    AttemptCommand,
    CreateGroup,
    DispatchTask,
    Recover,
    RequestGoalReview,
    StartGoal,
    StartTurn,
)
from milknado.domains.coordinator.control_services import TurnRuntimeHooks, TurnRuntimeRequest
from milknado.domains.coordinator.persistence import provider_bindings_for_session
from milknado.domains.coordinator.recovery import (
    ProviderIdentity,
    RecoveryOutcome,
    RecoveryRuntime,
)
from milknado.domains.graph import ExecutionGroup, MikadoGraph
from milknado.loop._agent import AgentResult
from milknado.loop.sessions import (
    ProviderSessionIdentity,
    RecoveryReceipt,
    RuntimeRequest,
    RuntimeResult,
)


class TurnRuntime:
    def __init__(self, graph: MikadoGraph, *, terminal: bool = True) -> None:
        self.graph: MikadoGraph = graph
        self.terminal: bool = terminal
        self.calls: list[ProviderSessionIdentity | None] = []

    def run(self, request: TurnRuntimeRequest) -> RuntimeResult:
        assert request.provider == "codex" and request.prompt == "Work"
        assert request.group is None or Path(request.group.worktree_path).is_absolute()
        acquired: list[bool] = []

        def probe_lock() -> None:
            with self.graph.synchronization_lock:
                acquired.append(True)

        probe = Thread(target=probe_lock)
        probe.start()
        probe.join(timeout=1)
        assert acquired == [True], "provider turn held the graph lock"
        self.calls.append(request.identity)
        provider_id = request.identity.session_id if request.identity else "provider-confirmed"
        request.hooks.identity(provider_id)
        recovery = (
            RecoveryReceipt(request.identity, "resumed", self.terminal)
            if request.identity
            else None
        )
        return RuntimeResult(
            AgentResult(0, session_id=provider_id, terminal_confirmed=self.terminal), recovery
        )


def _started(
    graph: MikadoGraph, root: Path, runtime: TurnRuntime
) -> tuple[CoordinatorControl, str, int]:
    control = CoordinatorControl(graph, root, CoordinatorServices(turn_runtime=runtime))
    receipt = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    result = cast(dict[str, object], receipt.result)
    return control, cast(str, result["id"]), cast(int, result["goal_id"])


def test_coordinator_first_turn_binds_provider_then_resumes_with_receipt(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    runtime = TurnRuntime(graph)
    control, session_id, _ = _started(graph, tmp_path, runtime)
    first = control.send_coordinator_command(session_id, StartTurn("turn-1", "Work"))
    assert first.status == "accepted"
    assert control.send_coordinator_command(session_id, StartTurn("turn-1", "Work")) == first
    assert runtime.calls == [None]
    second = control.send_coordinator_command(session_id, StartTurn("turn-2", "Work"))
    assert second.status == "accepted"
    assert runtime.calls[1] == ProviderSessionIdentity("codex", "provider-confirmed")
    snapshot = control.read_coordinator_snapshot(session_id, 0)
    assert [
        (item.scope_kind, item.provider_session_id) for item in snapshot.provider_bindings
    ] == [("coordinator", "provider-confirmed")]
    assert [(item.turn_id, item.status) for item in snapshot.provider_turns] == [
        ("turn-1", "confirmed"),
        ("turn-2", "confirmed"),
    ]
    assert [item.status for item in snapshot.recovery] == ["resumed"]
    graph.close()


def test_group_turn_requires_launched_writer_and_binds_only_confirmed_identity(
    tmp_path: Path,
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    runtime = TurnRuntime(graph)
    control, session_id, goal_id = _started(graph, tmp_path, runtime)
    task = graph.add_node("Implement", goal_id)
    created = control.send_coordinator_command(
        session_id, CreateGroup("group", "main", (task.id,), str(tmp_path / "worktree"), "branch")
    )
    assert created.status == "accepted"
    group_id = cast(str, cast(dict[str, object], created.result)["id"])
    group = graph.groups.get(group_id)
    assert group is not None and group.provider_session_id is None
    assert (
        control.send_coordinator_command(
            session_id, StartTurn("early", "Work", group_id=group_id)
        ).status
        == "rejected"
    )
    dispatch = control.send_coordinator_command(
        session_id, DispatchTask("dispatch", group_id, task.id, "run")
    )
    attempt = cast(dict[str, object], cast(dict[str, object], dispatch.result)["attempt"])
    attempt_id = cast(str, attempt["attempt_id"])
    turn = StartTurn(
        "turn", "Work", group_id=group_id, node_id=task.id, run_id="run", attempt_id=attempt_id
    )
    assert control.send_coordinator_command(session_id, turn).status == "rejected"
    assert (
        control.send_coordinator_command(
            session_id, AttemptCommand("launch", group_id, task.id, "run", attempt_id)
        ).status
        == "accepted"
    )
    run = graph.runs.get(attempt_id)
    assert run is not None and run["status"] == "running"
    assert (
        control.send_coordinator_command(
            session_id, AttemptCommand("launch-again", group_id, task.id, "run", attempt_id)
        ).status
        == "accepted"
    )
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute(
            "SELECT count(*) FROM runs WHERE run_id = ?", (attempt_id,)
        ).fetchone() == (1,)
    accepted = control.send_coordinator_command(
        session_id,
        StartTurn(
            "real-turn",
            "Work",
            group_id=group_id,
            node_id=task.id,
            run_id="run",
            attempt_id=attempt_id,
        ),
    )
    assert accepted.status == "accepted"
    group = graph.groups.get(group_id)
    assert group is not None and group.provider_session_id == "provider-confirmed"
    assert graph.groups.active_attempt(group_id) is not None
    with sqlite3.connect(graph.db_path) as conn:
        assert [
            (item.scope_id, item.provider_session_id)
            for item in provider_bindings_for_session(conn, session_id)
        ] == [(group_id, "provider-confirmed")]
    graph.close()


def test_unconfirmed_turn_stays_submitted_and_recovery_marks_unknown(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    runtime = TurnRuntime(graph, terminal=False)
    control, session_id, _ = _started(graph, tmp_path, runtime)
    first = control.send_coordinator_command(session_id, StartTurn("turn", "Work"))
    assert first.status == "unavailable"
    assert control.read_coordinator_snapshot(session_id, 0).provider_turns[0].status == "submitted"
    assert (
        control.send_coordinator_command(session_id, StartTurn("next", "Work")).status
        == "unavailable"
    )
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute(
            "SELECT state FROM coordinator_turn_launches ORDER BY rowid"
        ).fetchall() == [("unknown",), ("unknown",)]
    graph.close()
    reopened = MikadoGraph(tmp_path / "graph.db")

    class Provider:
        def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome:
            del identity, cwd
            return "unavailable"

    class Worktrees:
        def restore(self, group: ExecutionGroup) -> bool:
            del group
            return False

    recovered = CoordinatorControl(
        reopened,
        tmp_path,
        CoordinatorServices(
            recovery_runtime=RecoveryRuntime(reopened.groups, tmp_path, Provider(), Worktrees())
        ),
    ).send_coordinator_command(session_id, Recover("recover"))
    assert recovered.status == "accepted"
    result = cast(dict[str, object], recovered.result)
    assert cast(list[dict[str, object]], result["unknown_turns"])[0]["turn_id"] == "turn"
    reopened.close()


def test_native_turn_adapter_uses_configured_command_and_exact_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[RuntimeRequest] = []

    def execute(request: RuntimeRequest) -> RuntimeResult:
        requests.append(request)
        return RuntimeResult(AgentResult(0, session_id="thread", terminal_confirmed=True))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)
    adapter = NativeCoordinatorTurns(
        tmp_path,
        MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / "graph.db",
            agent_family="codex",
            execution_agent="codex exec --sandbox workspace-write",
        ),
    )
    hooks = TurnRuntimeHooks("turn", lambda _identity: None, lambda _event: None)
    _ = adapter.run(TurnRuntimeRequest("codex", "Work", None, None, hooks))
    identity = ProviderSessionIdentity("codex", "thread")
    _ = adapter.run(TurnRuntimeRequest("codex", "Work", None, identity, hooks))
    assert [request.spec.cmd for request in requests] == [
        ["codex", "exec", "--sandbox", "workspace-write"]
    ] * 2
    assert requests[0].resume is None
    assert requests[1].resume is not None and requests[1].resume.identity == identity
    assert requests[1].spec.cwd == tmp_path


def test_first_turn_rejects_provider_outside_coordinator_family(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    runtime = TurnRuntime(graph)
    control, session_id, _ = _started(graph, tmp_path, runtime)
    receipt = control.send_coordinator_command(
        session_id, StartTurn("wrong-family", "Work", provider="claude")
    )
    assert receipt.status == "rejected"
    assert runtime.calls == []
    assert control.read_coordinator_snapshot(session_id, 0).provider_bindings == ()
    graph.close()


def test_group_turn_obeys_pending_goal_review(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    runtime = TurnRuntime(graph)
    control, session_id, goal_id = _started(graph, tmp_path, runtime)
    task = graph.add_node("Implement", goal_id)
    created = control.send_coordinator_command(
        session_id, CreateGroup("group", "main", (task.id,), str(tmp_path / "worktree"), "branch")
    )
    group_id = cast(str, cast(dict[str, object], created.result)["id"])
    dispatch = control.send_coordinator_command(
        session_id, DispatchTask("dispatch", group_id, task.id, "run")
    )
    attempt = cast(dict[str, object], cast(dict[str, object], dispatch.result)["attempt"])
    attempt_id = cast(str, attempt["attempt_id"])
    assert (
        control.send_coordinator_command(
            session_id, AttemptCommand("launch", group_id, task.id, "run", attempt_id)
        ).status
        == "accepted"
    )
    assert (
        control.send_coordinator_command(
            session_id, RequestGoalReview("review", "rev", "evidence", "change", "agent")
        ).status
        == "accepted"
    )
    turn = StartTurn(
        "paused", "Work", group_id=group_id, node_id=task.id, run_id="run", attempt_id=attempt_id
    )
    assert control.send_coordinator_command(session_id, turn).status == "rejected"
    assert runtime.calls == []
    graph.close()


def test_first_native_turn_rejects_effective_cwd_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[RuntimeRequest] = []

    def capture(request: RuntimeRequest) -> RuntimeResult:
        requests.append(request)
        return RuntimeResult(AgentResult(0))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", capture)
    adapter = NativeCoordinatorTurns(
        tmp_path,
        MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / "graph.db",
            agent_family="codex",
            execution_agent=f"codex exec --cd {tmp_path.parent}",
        ),
    )
    with pytest.raises(ValueError, match="effective cwd"):
        _ = adapter.run(
            TurnRuntimeRequest(
                "codex",
                "Work",
                None,
                None,
                TurnRuntimeHooks("turn", lambda _identity: None, lambda _event: None),
            )
        )
    assert requests == []


def test_preflight_failure_releases_launch_fence_while_supervisor_lives(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    adapter = NativeCoordinatorTurns(
        tmp_path,
        MilknadoConfig(
            project_root=tmp_path,
            db_path=graph.db_path,
            agent_family="codex",
            execution_agent=f"codex exec --cd {tmp_path.parent}",
        ),
    )
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(turn_runtime=adapter, turn_owner=adapter.owner)
    )
    session = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], session.result)["id"])
    first = control.send_coordinator_command(session_id, StartTurn("first", "Work"))
    second = control.send_coordinator_command(session_id, StartTurn("second", "Work"))
    assert first.status == second.status == "unavailable"
    assert "effective cwd" in cast(str, second.result)
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute(
            "SELECT state FROM coordinator_turn_launches ORDER BY rowid"
        ).fetchall() == [("unknown",), ("unknown",)]
    graph.close()


def test_known_identity_has_submitted_evidence_before_provider_runs(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")

    class EvidenceRuntime(TurnRuntime):
        @override
        def run(self, request: TurnRuntimeRequest) -> RuntimeResult:
            if request.identity is not None:
                snapshot = control.read_coordinator_snapshot(session_id, 0)
                assert (
                    snapshot.provider_turns[-1].turn_id,
                    snapshot.provider_turns[-1].status,
                ) == ("resume", "submitted")
            return super().run(request)

    runtime = EvidenceRuntime(graph)
    control, session_id, _ = _started(graph, tmp_path, runtime)
    assert (
        control.send_coordinator_command(session_id, StartTurn("first", "Work")).status
        == "accepted"
    )
    assert (
        control.send_coordinator_command(session_id, StartTurn("resume", "Work")).status
        == "accepted"
    )
    graph.close()


def test_runtime_exception_keeps_launch_fenced_until_worker_is_verified(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")

    class FailedRuntime(TurnRuntime):
        @override
        def run(self, request: TurnRuntimeRequest) -> RuntimeResult:
            request.hooks.identity("thread")
            raise OSError("worker exit is not verified")

    control, session_id, _ = _started(graph, tmp_path, FailedRuntime(graph))
    receipt = control.send_coordinator_command(session_id, StartTurn("failed", "Work"))
    assert receipt.status == "unavailable"
    assert (
        control.send_coordinator_command(session_id, StartTurn("next", "Work")).status
        == "rejected"
    )
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute(
            "SELECT state FROM coordinator_turn_launches WHERE command_id = 'failed'"
        ).fetchone() == ("submitted",)
    graph.close()


class ScriptedRuntime:
    def __init__(self, script: Callable[[TurnRuntimeRequest], RuntimeResult]) -> None:
        self.script: Callable[[TurnRuntimeRequest], RuntimeResult] = script

    def run(self, request: TurnRuntimeRequest) -> RuntimeResult:
        return self.script(request)


def _scripted(
    tmp_path: Path, script: Callable[[TurnRuntimeRequest], RuntimeResult]
) -> tuple[MikadoGraph, CoordinatorControl, str]:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(turn_runtime=ScriptedRuntime(script))
    )
    receipt = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    return graph, control, cast(str, cast(dict[str, object], receipt.result)["id"])


def _confirmed(request: TurnRuntimeRequest) -> RuntimeResult:
    request.hooks.identity("provider")
    return RuntimeResult(AgentResult(0, session_id="provider", terminal_confirmed=True))


def test_turn_with_blank_prompt_is_rejected_before_launch(tmp_path: Path) -> None:
    graph, control, session_id = _scripted(tmp_path, _confirmed)
    receipt = control.send_coordinator_command(session_id, StartTurn("blank", "   "))
    assert (receipt.status, receipt.result) == ("rejected", "turn prompt must not be empty")
    assert control.read_coordinator_snapshot(session_id, 0).provider_turns == ()
    graph.close()


def test_turn_for_unknown_session_raises_key_error(tmp_path: Path) -> None:
    graph, control, _ = _scripted(tmp_path, _confirmed)
    with pytest.raises(KeyError, match="missing"):
        _ = control.send_coordinator_command("missing", StartTurn("turn", "Work"))
    graph.close()


def test_turn_without_runtime_is_unavailable(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path, CoordinatorServices())
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    receipt = control.send_coordinator_command(session_id, StartTurn("turn", "Work"))
    assert (receipt.status, receipt.result) == ("unavailable", "Turn runtime is not connected.")
    graph.close()


def test_turn_provider_must_match_existing_binding(tmp_path: Path) -> None:
    graph, control, session_id = _scripted(tmp_path, _confirmed)
    assert control.send_coordinator_command(session_id, StartTurn("first", "Work")).status == (
        "accepted"
    )
    receipt = control.send_coordinator_command(
        session_id, StartTurn("second", "Work", provider="claude")
    )
    assert (receipt.status, receipt.result) == (
        "rejected",
        "turn provider conflicts with existing binding",
    )
    graph.close()


def test_empty_provider_identity_is_unavailable_and_binds_nothing(tmp_path: Path) -> None:
    def blank_identity(request: TurnRuntimeRequest) -> RuntimeResult:
        request.hooks.identity("")
        return RuntimeResult(None)

    graph, control, session_id = _scripted(tmp_path, blank_identity)
    receipt = control.send_coordinator_command(session_id, StartTurn("turn", "Work"))
    assert (receipt.status, receipt.result) == (
        "unavailable",
        "provider did not confirm session identity",
    )
    assert control.read_coordinator_snapshot(session_id, 0).provider_bindings == ()
    graph.close()


def test_permission_event_before_identity_is_unavailable(tmp_path: Path) -> None:
    def early_permission(request: TurnRuntimeRequest) -> RuntimeResult:
        request.hooks.event(SessionEvent(kind="permission", text="Approve", event_id="ask"))
        return RuntimeResult(None)

    graph, control, session_id = _scripted(tmp_path, early_permission)
    receipt = control.send_coordinator_command(session_id, StartTurn("turn", "Work"))
    assert (receipt.status, receipt.result) == (
        "unavailable",
        "permission event has no confirmed provider identity",
    )
    graph.close()


def test_turn_without_run_result_is_unavailable(tmp_path: Path) -> None:
    graph, control, session_id = _scripted(tmp_path, lambda _request: RuntimeResult(None))
    receipt = control.send_coordinator_command(session_id, StartTurn("turn", "Work"))
    assert (receipt.status, receipt.result) == (
        "unavailable",
        "Provider did not confirm a session identity.",
    )
    graph.close()


def test_resumed_turn_must_keep_bound_session(tmp_path: Path) -> None:
    def switching(request: TurnRuntimeRequest) -> RuntimeResult:
        if request.identity is None:
            return _confirmed(request)
        return RuntimeResult(AgentResult(0, session_id="other", terminal_confirmed=True))

    graph, control, session_id = _scripted(tmp_path, switching)
    assert control.send_coordinator_command(session_id, StartTurn("first", "Work")).status == (
        "accepted"
    )
    receipt = control.send_coordinator_command(session_id, StartTurn("second", "Work"))
    assert (receipt.status, receipt.result) == (
        "unavailable",
        "Provider resumed a different session.",
    )
    graph.close()


def test_resumed_turn_cannot_confirm_a_different_identity(tmp_path: Path) -> None:
    def switching(request: TurnRuntimeRequest) -> RuntimeResult:
        if request.identity is None:
            return _confirmed(request)
        request.hooks.identity("other")
        return RuntimeResult(None)

    graph, control, session_id = _scripted(tmp_path, switching)
    _ = control.send_coordinator_command(session_id, StartTurn("first", "Work"))
    receipt = control.send_coordinator_command(session_id, StartTurn("second", "Work"))
    assert (receipt.status, receipt.result) == (
        "unavailable",
        "provider resumed a different session",
    )
    graph.close()
