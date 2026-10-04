from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import CONTROLLER_MASTER_ENV
from milknado.domains.coordinator import ControlEvent, CoordinatorControl, ProviderBinding
from milknado.domains.coordinator import projection as coordinator_projection
from milknado.domains.coordinator.control_models import (
    DecideGoalReview,
    PlanGoal,
    Recover,
    RequestGoalReview,
    StartGoal,
)
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.coordinator.journal import append_control_event
from milknado.domains.coordinator.persistence import bind_provider_session, link_entity
from milknado.domains.coordinator.recovery import (
    ProviderIdentity,
    RecoveryOutcome,
    RecoveryRuntime,
)
from milknado.domains.graph import ExecutionGroup, GoalReviewDecision, MikadoGraph
from milknado.domains.planning import Planner, PlanResult


def _session(control: CoordinatorControl) -> tuple[str, int]:
    receipt = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    result = cast(dict[str, object], receipt.result)
    return cast(str, result["id"]), cast(int, result["goal_id"])


def _review(review_id: str = "review", reviewer: str = "agent") -> RequestGoalReview:
    return RequestGoalReview(review_id, "rev-1", "new evidence", "change goal", reviewer)


def test_review_requires_identity_and_projects_decision_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "review-secret")
    graph = MikadoGraph(tmp_path / "graph.db")
    graph.register_controller_master()
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(review_decision=graph.decide_goal_review)
    )
    session_id, _ = _session(control)
    invalid = control.send_coordinator_command(session_id, _review("invalid", " "))
    assert invalid.status == "rejected"
    assert control.read_coordinator_snapshot(session_id, 0).reviews == ()
    pending = control.send_coordinator_command(session_id, _review())
    assert pending.status == "accepted"
    review_id = cast(int, cast(dict[str, object], pending.result)["review_id"])
    first = control.read_coordinator_snapshot(session_id, 0)
    assert first.reviews[0].reviewer == "agent"
    assert first.reviews[0].decision.value == "pending"
    accepted = control.send_coordinator_command(
        session_id, DecideGoalReview("decide", review_id, GoalReviewDecision.ACCEPTED, "human")
    )
    assert accepted.status == "accepted"
    after = control.read_coordinator_snapshot(session_id, first.cursor)
    assert after.reviews[0].decision.value == "accepted"
    assert [(event.kind, event.status) for event in after.events] == [
        ("approval", "accepted"),
        ("command", "accepted"),
    ]
    assert after.cursor > first.cursor
    graph.close()


def test_denied_decision_is_durable_not_unconfirmed(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    session_id, _ = _session(control)
    pending = control.send_coordinator_command(session_id, _review())
    review_id = cast(int, cast(dict[str, object], pending.result)["review_id"])
    command = DecideGoalReview("decide", review_id, GoalReviewDecision.REJECTED, "human")
    unavailable = control.send_coordinator_command(session_id, command)
    assert unavailable.status == "unavailable"
    assert control.send_coordinator_command(session_id, command) == unavailable
    graph.close()


def test_uncredentialed_decision_is_rejected_and_replayed(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(review_decision=graph.decide_goal_review)
    )
    session_id, _ = _session(control)
    pending = control.send_coordinator_command(session_id, _review())
    review_id = cast(int, cast(dict[str, object], pending.result)["review_id"])
    command = DecideGoalReview("decide", review_id, GoalReviewDecision.ACCEPTED, "human")
    denied = control.send_coordinator_command(session_id, command)
    assert denied.status == "rejected"
    assert control.send_coordinator_command(session_id, command) == denied
    assert control.read_coordinator_snapshot(session_id, 0).reviews[0].decision.value == "pending"
    graph.close()


def test_projection_cursor_matches_node_state_across_external_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    session_id, goal_id = _session(control)
    original = coordinator_projection._goal_nodes  # pyright: ignore[reportPrivateUsage]

    def interleave(owner: MikadoGraph, node_id: int):
        with sqlite3.connect(graph.db_path) as writer:
            _ = writer.execute(
                "UPDATE nodes SET description = ? WHERE id = ?", ("Changed", goal_id)
            )
            _ = append_control_event(
                writer,
                session_id,
                ControlEvent(
                    kind="command", entity_kind="test", entity_id="change", status="applied"
                ),
            )
        return original(owner, node_id)

    monkeypatch.setattr(coordinator_projection, "_goal_nodes", interleave)
    before = control.read_coordinator_snapshot(session_id, 0)
    assert before.goal.description == "Deliver"
    assert [(event.entity_id, event.status) for event in before.events] == [("start", "accepted")]
    monkeypatch.setattr(coordinator_projection, "_goal_nodes", original)
    after = control.read_coordinator_snapshot(session_id, before.cursor)
    assert after.goal.description == "Changed"
    assert [event.entity_id for event in after.events] == ["change"]
    graph.close()


def test_plan_receipt_serializes_context_path_and_replays(tmp_path: Path) -> None:
    class PlannerStub:
        def launch(
            self, goal: str, project_root: Path, *, target_goal_id: int | None = None
        ) -> PlanResult:
            assert goal == "Deliver"
            assert project_root == tmp_path
            assert target_goal_id is not None
            return PlanResult(True, 0, tmp_path / "context.md", nodes_created=2)

    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(planner=cast(Planner, cast(object, PlannerStub())))
    )
    session_id, _ = _session(control)
    command = PlanGoal("plan")
    first = control.send_coordinator_command(session_id, command)
    assert first.status == "accepted"
    assert cast(dict[str, object], first.result)["context_path"] == str(tmp_path / "context.md")
    assert control.send_coordinator_command(session_id, command) == first
    graph.close()


def test_recovery_receipt_serializes_worktree_path_and_replays(tmp_path: Path) -> None:
    class ProviderStub:
        def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome:
            assert identity == ProviderIdentity("codex", "provider-1")
            assert cwd == tmp_path
            return "resumed"

    class WorktreeStub:
        def restore(self, group: ExecutionGroup) -> bool:
            raise AssertionError(f"No group needs restoration: {group.id}")

    graph = MikadoGraph(tmp_path / "graph.db")
    recovery = RecoveryRuntime(graph.groups, tmp_path, ProviderStub(), WorktreeStub())
    control = CoordinatorControl(graph, tmp_path, CoordinatorServices(recovery_runtime=recovery))
    session_id, _ = _session(control)
    link_entity(graph.group_connection, session_id, "provider_session", "provider-1")
    bind_provider_session(
        graph.group_connection,
        session_id,
        ProviderBinding("coordinator", session_id, "codex", "provider-1"),
    )
    command = Recover("recover")
    first = control.send_coordinator_command(session_id, command)
    assert first.status == "accepted"
    receipts = cast(list[dict[str, object]], cast(dict[str, object], first.result)["receipts"])
    assert receipts[0]["worktree_path"] == str(tmp_path)
    assert receipts[0]["outcome"] == "resumed"
    assert control.send_coordinator_command(session_id, command) == first
    graph.close()


def test_command_receipts_append_one_redacted_event_per_outcome(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    start = control.send_coordinator_command("", StartGoal("start", "password=private", "codex"))
    session_id = cast(str, cast(dict[str, object], start.result)["id"])
    unavailable = Recover("recover-secret")
    rejected = RequestGoalReview("reject-secret", "rev", "secret=private", "change", " ")
    assert control.send_coordinator_command(session_id, unavailable).status == "unavailable"
    assert control.send_coordinator_command(session_id, rejected).status == "rejected"
    assert control.send_coordinator_command(session_id, unavailable).status == "unavailable"
    events = [
        event
        for event in control.read_coordinator_snapshot(session_id, 0).events
        if event.kind == "command"
    ]
    assert [(event.entity_id, event.status) for event in events] == [
        ("start", "accepted"),
        ("recover-secret", "unavailable"),
        ("reject-secret", "rejected"),
    ]
    assert all("private" not in event.text for event in events)
    graph.close()


def test_fresh_database_missing_session_raises_key_error(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    with pytest.raises(KeyError):
        _ = control.read_coordinator_snapshot("missing", 0)
    with pytest.raises(KeyError):
        _ = control.send_coordinator_command("missing", Recover("recover"))
    graph.close()
