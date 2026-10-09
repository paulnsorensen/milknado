from __future__ import annotations

from functools import partial
from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest

from milknado.domains.common import CONTROLLER_MASTER_ENV
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator._command_lifecycle import CommandLifecycle
from milknado.domains.coordinator.control_models import RequestGoalReview, StartGoal, StartTurn
from milknado.domains.coordinator.control_services import (
    TurnRunResult,
    TurnRuntimeRequest,
    TurnRuntimeResult,
)
from milknado.domains.graph import (
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    MikadoGraph,
)


class _Runtime:
    def __init__(self) -> None:
        self.returned: Event = Event()

    def run(self, request: TurnRuntimeRequest) -> TurnRuntimeResult:
        _ = request
        self.returned.set()
        return TurnRuntimeResult(TurnRunResult("native-session", True))


def _started(root: Path) -> tuple[MikadoGraph, CoordinatorControl, str, _Runtime]:
    graph = MikadoGraph(root / "graph.db")
    runtime = _Runtime()
    control = CoordinatorControl(graph, root, CoordinatorServices(turn_runtime=runtime))
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    return graph, control, session_id, runtime


def _assert_final_receipt(graph: MikadoGraph) -> None:
    row = cast(
        tuple[str, str, str] | None,
        graph.group_connection.execute(
            "SELECT receipt.status, receipt.result_json, launch.state "
            + "FROM coordinator_web_receipts AS receipt "
            + "JOIN coordinator_turn_launches AS launch USING (command_id) "
            + "WHERE command_id = ?",
            ("turn",),
        ).fetchone(),
    )
    assert row is not None and row[:3] == (
        "accepted",
        '{"provider_session_id":"native-session","turn_id":"turn"}',
        "confirmed",
    )


def test_shutdown_waits_for_final_turn_receipt_and_closes_command_admission(
    tmp_path: Path,
) -> None:
    graph, control, session_id, runtime = _started(tmp_path)
    finalizing, release, closing, stopped = Event(), Event(), Event(), Event()

    def trace(sql: str) -> None:
        if "UPDATE coordinator_web_receipts SET status" in sql:
            finalizing.set()
            _ = release.wait(5)

    graph.group_connection.set_trace_callback(trace)
    receipts: list[str] = []
    turn = Thread(
        target=lambda: receipts.append(
            control.send_coordinator_command(session_id, StartTurn("turn", "Work")).status
        )
    )
    turn.start()
    assert runtime.returned.wait(5) and finalizing.wait(5)
    shutdown = Thread(target=lambda: (control.shutdown(closing.set), stopped.set()))
    shutdown.start()
    assert closing.wait(5)
    assert not stopped.is_set()
    with pytest.raises(RuntimeError, match="shutting down"):
        _ = control.send_coordinator_command(session_id, StartTurn("late", "Work"))
    release.set()
    turn.join(5)
    shutdown.join(5)
    assert stopped.is_set() and receipts == ["accepted"]
    graph.group_connection.set_trace_callback(None)
    _assert_final_receipt(graph)
    graph.close()


def test_shutdown_reports_unconfirmed_command_after_deadline(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    lifecycle = CommandLifecycle(lambda: graph.group_connection)
    started, release = Event(), Event()

    def command() -> None:
        with lifecycle.command():
            started.set()
            assert release.wait(5)

    active = Thread(target=command)
    active.start()
    assert started.wait(5)
    with pytest.raises(RuntimeError, match="coordinator command shutdown is unconfirmed"):
        lifecycle.shutdown(lambda: None, timeout=0.01)
    release.set()
    active.join(5)
    assert not active.is_alive()
    graph.close()


def test_shutdown_waits_for_public_review_decision_and_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "review-secret")
    graph = MikadoGraph(tmp_path / "graph.db")
    graph.register_controller_master()
    entered, release, closing, stopped = Event(), Event(), Event(), Event()

    def decide(request: GoalReviewDecisionRequest, *, decided_by: str) -> GoalReviewRecord:
        entered.set()
        assert release.wait(5)
        return graph.decide_goal_review(request, decided_by=decided_by)

    control = CoordinatorControl(graph, tmp_path, CoordinatorServices(review_decision=decide))
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    pending = control.send_coordinator_command(
        session_id, RequestGoalReview("review", "rev-1", "evidence", "change", "agent")
    )
    review_id = cast(int, cast(dict[str, object], pending.result)["review_id"])
    before = control.read_coordinator_snapshot(session_id, 0)
    request = GoalReviewDecisionRequest(review_id, GoalReviewDecision.ACCEPTED)
    review = Thread(target=partial(control.decide_goal_review, request, decided_by="human"))
    review.start()
    assert entered.wait(5)
    shutdown = Thread(target=lambda: (control.shutdown(closing.set), stopped.set()))
    shutdown.start()
    assert closing.wait(5) and not stopped.is_set()
    with pytest.raises(RuntimeError, match="shutting down"):
        _ = control.decide_goal_review(request, decided_by="late")
    release.set()
    review.join(5)
    shutdown.join(5)
    assert stopped.is_set()
    after = control.read_coordinator_snapshot(session_id, before.cursor)
    assert [(event.kind, event.status) for event in after.events] == [("approval", "accepted")]
    graph.close()
