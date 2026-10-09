from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from threading import Thread
from typing import cast

import psutil
import pytest

from milknado.adapters.coordinator_worker_recovery import CoordinatorWorkerRecovery
from milknado.domains.common import WorkerIdentity, WorkerOwner
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator.control_models import Recover, StartGoal, StartTurn
from milknado.domains.coordinator.control_services import (
    TurnRunResult,
    TurnRuntimeRequest,
    TurnRuntimeResult,
)
from milknado.domains.coordinator.recovery import (
    ProviderIdentity,
    RecoveryOutcome,
    RecoveryRuntime,
)
from milknado.domains.graph import ExecutionGroup, MikadoGraph, WorkerEvidenceStore


class Runtime:
    def run(self, request: TurnRuntimeRequest) -> TurnRuntimeResult:
        request.hooks.identity("thread")
        return TurnRuntimeResult(TurnRunResult("thread", False))


class Provider:
    def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome:
        del identity, cwd
        return "unavailable"


class Worktrees:
    def restore(self, group: ExecutionGroup) -> bool:
        del group
        return False


class Workers:
    def __init__(self, terminated: bool) -> None:
        self.result: bool = terminated
        self.turns: list[str] = []

    def terminated(self, turn_id: str, supervisor_pid: int, supervisor_start_token: float) -> bool:
        assert (supervisor_pid, supervisor_start_token) == (999999, 1.0)
        self.turns.append(turn_id)
        return self.result


def test_recovery_keeps_unknown_turn_fenced_until_worker_termination(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")

    def owner(turn_id: str) -> WorkerOwner:
        return WorkerOwner(turn_id, 999999, 1.0, None, None)

    first = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(turn_runtime=Runtime(), turn_owner=owner)
    )
    goal = first.send_coordinator_command("", StartGoal("goal", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], goal.result)["id"])
    assert (
        first.send_coordinator_command(session_id, StartTurn("turn", "Work")).status
        == "unavailable"
    )
    with sqlite3.connect(graph.db_path) as conn:
        _ = conn.execute("UPDATE coordinator_turn_launches SET state = 'submitted'")
    graph.close()

    reopened = MikadoGraph(tmp_path / "graph.db")
    workers = Workers(False)
    control = CoordinatorControl(
        reopened,
        tmp_path,
        CoordinatorServices(
            turn_runtime=Runtime(),
            turn_owner=owner,
            recovery_runtime=RecoveryRuntime(
                reopened.groups, tmp_path, Provider(), Worktrees(), workers
            ),
        ),
    )
    assert control.send_coordinator_command(session_id, Recover("check-live")).status == "accepted"
    assert workers.turns == ["turn"]
    assert (
        control.send_coordinator_command(session_id, StartTurn("blocked", "Work")).status
        == "rejected"
    )
    workers.result = True
    assert control.send_coordinator_command(session_id, Recover("check-dead")).status == "accepted"
    assert workers.turns == ["turn", "turn"]
    assert (
        control.send_coordinator_command(session_id, StartTurn("next", "Work")).status
        == "unavailable"
    )
    assert [
        (turn.turn_id, turn.status)
        for turn in control.read_coordinator_snapshot(session_id, 0).provider_turns
    ] == [("turn", "unknown"), ("next", "submitted")]
    reopened.close()


def test_recovery_does_not_clear_new_launch_created_during_worker_check(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")

    def owner(turn_id: str) -> WorkerOwner:
        return WorkerOwner(turn_id, 999999, 1.0, None, None)

    class RacingWorkers:
        def terminated(
            self, turn_id: str, supervisor_pid: int, supervisor_start_token: float
        ) -> bool:
            del supervisor_pid, supervisor_start_token
            acquired: list[bool] = []

            def probe_lock() -> None:
                with graph.synchronization_lock:
                    acquired.append(True)

            probe = Thread(target=probe_lock)
            probe.start()
            probe.join(timeout=1)
            assert acquired == [True], "worker check held the graph lock"
            with sqlite3.connect(graph.db_path) as conn:
                row = cast(
                    tuple[str, str, str] | None,
                    conn.execute(
                        "SELECT coordinator_id, scope_kind, scope_id "
                        + "FROM coordinator_turn_launches WHERE command_id = ?",
                        (turn_id,),
                    ).fetchone(),
                )
                assert row is not None
                _ = conn.execute(
                    "UPDATE coordinator_turn_launches SET state = 'unknown' WHERE command_id = ?",
                    (turn_id,),
                )
                _ = conn.execute(
                    "INSERT INTO coordinator_turn_launches "
                    + "(command_id, coordinator_id, scope_kind, scope_id, state, "
                    + "supervisor_pid, supervisor_start_token) "
                    + "VALUES ('new', ?, ?, ?, 'submitted', 999999, 1.0)",
                    row,
                )
            return True

    runtime = Runtime()
    start = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(turn_runtime=runtime, turn_owner=owner)
    )
    goal = start.send_coordinator_command("", StartGoal("goal", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], goal.result)["id"])
    assert (
        start.send_coordinator_command(session_id, StartTurn("old", "Work")).status
        == "unavailable"
    )
    with sqlite3.connect(graph.db_path) as conn:
        _ = conn.execute("UPDATE coordinator_turn_launches SET state = 'submitted'")
    recover = CoordinatorControl(
        graph,
        tmp_path,
        CoordinatorServices(
            turn_runtime=runtime,
            turn_owner=owner,
            recovery_runtime=RecoveryRuntime(
                graph.groups, tmp_path, Provider(), Worktrees(), RacingWorkers()
            ),
        ),
    )
    assert recover.send_coordinator_command(session_id, Recover("race")).status == "accepted"
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute(
            "SELECT command_id, state FROM coordinator_turn_launches ORDER BY command_id"
        ).fetchall() == [("new", "submitted"), ("old", "unknown")]
    graph.close()


def test_worker_recovery_requires_supervisor_exit(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    verifier = CoordinatorWorkerRecovery(graph.db_path)
    assert not verifier.terminated("turn", os.getpid(), psutil.Process().create_time())
    assert verifier.terminated("turn", os.getpid(), 0.0)
    graph.close()


def test_worker_recovery_selects_exact_runtime_turn_with_group_association(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("Worker")
    graph.runs.start("graph-run", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    for turn_id, invocation in (("turn", "matching"), ("other", "unrelated")):
        graph.runs.record_worker(
            WorkerOwner(turn_id, 999999, 1.0, "graph-run", node.id),
            WorkerIdentity(invocation, 2345, 2345, 1.0),
        )
    verified: list[str] = []

    def terminate(identity: WorkerIdentity, descendants: object, deadline: float) -> bool:
        del descendants, deadline
        verified.append(identity.invocation_id)
        return False

    monkeypatch.setattr(
        "milknado.adapters.coordinator_worker_recovery.terminate_verified", terminate
    )
    verifier = CoordinatorWorkerRecovery(graph.db_path)
    assert verifier.terminated("turn", 999999, 1.0)
    assert verified == ["matching"]
    with WorkerEvidenceStore(graph.db_path) as store:
        matching = store.get("matching")
        unrelated = store.get("unrelated")
        assert matching is not None and matching.ended_at is not None
        assert unrelated is not None and unrelated.ended_at is None
    graph.close()


def _graph_with_worker(tmp_path: Path, supervisor: tuple[int, float]) -> MikadoGraph:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("Worker")
    graph.runs.start("graph-run", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    graph.runs.record_worker(
        WorkerOwner("turn", supervisor[0], supervisor[1], "graph-run", node.id),
        WorkerIdentity("worker", 2345, 2345, 1.0),
    )
    return graph


def _termination_result(monkeypatch: pytest.MonkeyPatch, survived: bool) -> None:
    def terminate(identity: WorkerIdentity, descendants: object, deadline: float) -> bool:
        del identity, descendants, deadline
        return survived

    monkeypatch.setattr(
        "milknado.adapters.coordinator_worker_recovery.terminate_verified", terminate
    )


def test_worker_recovery_refuses_worker_owned_by_another_supervisor(tmp_path: Path) -> None:
    graph = _graph_with_worker(tmp_path, (888888, 2.0))
    assert not CoordinatorWorkerRecovery(graph.db_path).terminated("turn", 999999, 1.0)
    with WorkerEvidenceStore(graph.db_path) as store:
        worker = store.get("worker")
        assert worker is not None and worker.ended_at is None
    graph.close()


def test_worker_recovery_keeps_fence_when_worker_survives_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph_with_worker(tmp_path, (999999, 1.0))
    _termination_result(monkeypatch, True)
    assert not CoordinatorWorkerRecovery(graph.db_path).terminated("turn", 999999, 1.0)
    with WorkerEvidenceStore(graph.db_path) as store:
        worker = store.get("worker")
        assert worker is not None and worker.ended_at is None
    graph.close()


def test_worker_recovery_keeps_fence_when_evidence_cannot_be_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph_with_worker(tmp_path, (999999, 1.0))

    def refuse(*_args: object) -> None:
        raise RuntimeError("stale snapshot")

    _termination_result(monkeypatch, False)
    monkeypatch.setattr(WorkerEvidenceStore, "end", refuse)
    assert not CoordinatorWorkerRecovery(graph.db_path).terminated("turn", 999999, 1.0)
    graph.close()
