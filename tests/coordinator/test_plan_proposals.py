from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from threading import Event
from typing import cast

import pytest
from typing_extensions import override

from milknado.domains.batching import Batch, BatchPlan
from milknado.domains.common import MikadoNode
from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import DecidePlanProposal, PlanGoal, StartGoal
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.coordinator.planning_workflow import CoordinatorPlanning
from milknado.domains.graph import MikadoGraph, graph_revision
from milknado.domains.planning import (
    Planner,
    PlanProposal,
    PlanResult,
    apply_batches_to_graph,
    decode_manifest,
)


class _PreparedStub:
    def prepare_proposal(self, proposal: PlanProposal, project_root: Path) -> BatchPlan:
        _ = (proposal, project_root)
        return BatchPlan((), (), "OPTIMAL")


def _planner(graph: MikadoGraph, root: Path) -> Planner:
    class Stub(_PreparedStub):
        applied: int = 0

        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            assert (goal, project_root) == ("Deliver", root)
            assert graph.get_children(target_goal_id) == []
            manifest = decode_manifest(
                {
                    "manifest_version": "milknado.plan.v2",
                    "goal": "Deliver",
                    "goal_summary": "Deliver",
                    "changes": [
                        {"id": "task-1", "path": "src/a.py", "description": "Implement task"}
                    ],
                    "new_relationships": [],
                }
            )
            return PlanProposal(manifest, root / "context.md")

        def apply_proposal(
            self,
            proposal: PlanProposal,
            *,
            target_goal_id: int,
            prepared_plan: BatchPlan,
        ) -> PlanResult:
            _ = prepared_plan
            self.applied += 1
            _ = graph.add_node("Implement task", target_goal_id)
            return PlanResult(True, 0, proposal.context_path, nodes_created=1)

    return cast(Planner, cast(object, Stub()))


def _start(control: CoordinatorControl) -> tuple[str, int]:
    receipt = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    result = cast(dict[str, object], receipt.result)
    return cast(str, result["id"]), cast(int, result["goal_id"])


def test_proposal_requires_review_and_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    planner = _planner(graph, tmp_path)
    control = CoordinatorControl(graph, tmp_path, CoordinatorServices(planner=planner))
    session_id, goal_id = _start(control)
    receipt = control.send_coordinator_command(session_id, PlanGoal("proposal-1"))
    assert receipt.status == "accepted"
    assert graph.get_children(goal_id) == []
    proposal = control.read_coordinator_snapshot(session_id, 0).proposals[0]
    assert proposal.status == "pending"
    assert proposal.manifest.changes[0].description == "Implement task"
    graph.close()

    reopened = MikadoGraph(path)
    planner = _planner(reopened, tmp_path)
    control = CoordinatorControl(reopened, tmp_path, CoordinatorServices(planner=planner))
    assert control.read_coordinator_snapshot(session_id, 0).proposals[0] == proposal
    approved = control.send_coordinator_command(
        session_id, DecidePlanProposal("approve-1", "proposal-1", "accepted")
    )
    assert approved.status == "accepted"
    assert len(reopened.get_children(goal_id)) == 1
    again = control.send_coordinator_command(
        session_id, DecidePlanProposal("approve-2", "proposal-1", "accepted")
    )
    assert again.status == "accepted"
    assert len(reopened.get_children(goal_id)) == 1
    reopened.close()


def test_applied_retry_repairs_missing_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(planner=_planner(graph, tmp_path))
    )
    session_id, goal_id = _start(control)
    _ = control.send_coordinator_command(session_id, PlanGoal("proposal"))
    original = CoordinatorPlanning.record_plan

    def crash_after_apply(
        self: CoordinatorPlanning, session: object, plan_id: str, status: str
    ) -> None:
        _ = (self, session, plan_id, status)
        raise RuntimeError("history write interrupted")

    monkeypatch.setattr(CoordinatorPlanning, "record_plan", crash_after_apply)
    with pytest.raises(RuntimeError, match="history write interrupted"):
        _ = control.send_coordinator_command(
            session_id, DecidePlanProposal("first-approval", "proposal", "accepted")
        )
    monkeypatch.setattr(CoordinatorPlanning, "record_plan", original)
    assert control.read_coordinator_snapshot(session_id, 0).proposals[0].status == "applied"
    assert len(graph.get_children(goal_id)) == 1
    retry = control.send_coordinator_command(
        session_id, DecidePlanProposal("retry-approval", "proposal", "accepted")
    )
    assert retry.status == "accepted"
    snapshot = control.read_coordinator_snapshot(session_id, 0)
    assert [
        (link.kind, link.entity_id) for link in snapshot.links if link.kind == "planning_decision"
    ] == [("planning_decision", "proposal")]
    assert [
        (event.entity_id, event.status)
        for event in snapshot.events
        if event.kind == "planning_decision"
    ] == [("proposal", "accepted")]
    assert len(graph.get_children(goal_id)) == 1
    graph.close()


def test_rejection_and_stale_graph_never_apply(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(planner=_planner(graph, tmp_path))
    )
    session_id, goal_id = _start(control)
    _ = control.send_coordinator_command(session_id, PlanGoal("reject-plan"))
    rejected = control.send_coordinator_command(
        session_id, DecidePlanProposal("reject", "reject-plan", "rejected")
    )
    assert rejected.status == "accepted"
    assert graph.get_children(goal_id) == []
    _ = control.send_coordinator_command(session_id, PlanGoal("stale-plan"))
    _ = graph.add_node("External change", goal_id)
    stale = control.send_coordinator_command(
        session_id, DecidePlanProposal("stale", "stale-plan", "accepted")
    )
    assert stale.status == "rejected"
    assert [node.description for node in graph.get_children(goal_id)] == ["External change"]
    assert control.read_coordinator_snapshot(session_id, 0).proposals[-1].status == "stale"
    graph.close()


def test_proposal_rejects_goal_update_between_input_read_and_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(planner=_planner(graph, tmp_path))
    )
    session_id, goal_id = _start(control)
    read_node = graph.get_node

    def read_then_update(node_id: int) -> MikadoNode | None:
        node = read_node(node_id)
        with sqlite3.connect(path) as other:
            _ = other.execute(
                "UPDATE nodes SET description = ? WHERE id = ?", ("Changed", goal_id)
            )
        return node

    monkeypatch.setattr(graph, "get_node", read_then_update)
    receipt = control.send_coordinator_command(session_id, PlanGoal("changed-goal"))
    monkeypatch.setattr(graph, "get_node", read_node)
    assert receipt.status == "rejected"
    assert "graph changed" in str(receipt.result)
    assert control.read_coordinator_snapshot(session_id, 0).proposals == ()
    changed = graph.get_node(goal_id)
    assert changed is not None and changed.description == "Changed"
    graph.close()


def test_proposal_rejects_graph_change_during_planning(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    delegate = _planner(graph, tmp_path)

    class ChangingPlanner:
        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            proposal = delegate.propose(goal, project_root, target_goal_id=target_goal_id)
            _ = graph.add_node("Concurrent change", target_goal_id)
            return proposal

    control = CoordinatorControl(
        graph,
        tmp_path,
        CoordinatorServices(planner=cast(Planner, cast(object, ChangingPlanner()))),
    )
    session_id, goal_id = _start(control)
    receipt = control.send_coordinator_command(session_id, PlanGoal("changed-during-plan"))
    assert receipt.status == "rejected"
    assert "graph changed" in str(receipt.result)
    assert control.read_coordinator_snapshot(session_id, 0).proposals == ()
    assert [node.description for node in graph.get_children(goal_id)] == ["Concurrent change"]
    graph.close()


def test_interrupted_apply_fails_closed_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    delegate = _planner(graph, tmp_path)

    class CrashPlanner(_PreparedStub):
        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            return delegate.propose(goal, project_root, target_goal_id=target_goal_id)

        def apply_proposal(
            self,
            proposal: PlanProposal,
            *,
            target_goal_id: int,
            prepared_plan: BatchPlan,
        ) -> PlanResult:
            _ = (proposal, prepared_plan)
            _ = graph.add_node("Partial task", target_goal_id)
            raise RuntimeError("worker stopped during apply")

    control = CoordinatorControl(
        graph,
        tmp_path,
        CoordinatorServices(planner=cast(Planner, cast(object, CrashPlanner()))),
    )
    session_id, goal_id = _start(control)
    _ = control.send_coordinator_command(session_id, PlanGoal("interrupted"))
    with pytest.raises(RuntimeError, match="worker stopped"):
        _ = control.send_coordinator_command(
            session_id, DecidePlanProposal("approve", "interrupted", "accepted")
        )
    assert control.read_coordinator_snapshot(session_id, 0).proposals[0].status == "applying"
    graph.close()

    reopened = MikadoGraph(path)
    control = CoordinatorControl(
        reopened, tmp_path, CoordinatorServices(planner=_planner(reopened, tmp_path))
    )
    result = control.send_coordinator_command(
        session_id, DecidePlanProposal("retry", "interrupted", "accepted")
    )
    assert result.status == "rejected"
    assert "manual recovery" in str(result.result)
    assert reopened.get_children(goal_id) == []
    reopened.close()


def test_competing_approvals_on_separate_connections_apply_once(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    entered, release = Event(), Event()
    manifest = decode_manifest(
        {
            "manifest_version": "milknado.plan.v2",
            "goal": "Deliver",
            "goal_summary": "Deliver",
            "changes": [],
            "new_relationships": [],
        }
    )

    class FirstPlanner(_PreparedStub):
        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            _ = (goal, project_root, target_goal_id)
            return PlanProposal(manifest, tmp_path / "context.md")

        def apply_proposal(
            self,
            proposal: PlanProposal,
            *,
            target_goal_id: int,
            prepared_plan: BatchPlan,
        ) -> PlanResult:
            _ = (target_goal_id, prepared_plan)
            entered.set()
            assert release.wait(5)
            return PlanResult(True, 0, proposal.context_path)

    class SecondPlanner(_PreparedStub):
        applied: bool = False

        def apply_proposal(
            self,
            proposal: PlanProposal,
            *,
            target_goal_id: int,
            prepared_plan: BatchPlan,
        ) -> PlanResult:
            _ = (target_goal_id, prepared_plan)
            self.applied = True
            return PlanResult(True, 0, proposal.context_path)

    first = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(planner=cast(Planner, cast(object, FirstPlanner())))
    )
    session_id, goal_id = _start(first)
    _ = first.send_coordinator_command(session_id, PlanGoal("first"))
    _ = first.send_coordinator_command(session_id, PlanGoal("second"))
    other_graph = MikadoGraph(graph.db_path)
    second_planner = SecondPlanner()
    second = CoordinatorControl(
        other_graph,
        tmp_path,
        CoordinatorServices(planner=cast(Planner, cast(object, second_planner))),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(
            first.send_coordinator_command,
            session_id,
            DecidePlanProposal("approve-first", "first", "accepted"),
        )
        assert entered.wait(5)
        second_result = pool.submit(
            second.send_coordinator_command,
            session_id,
            DecidePlanProposal("approve-second", "second", "accepted"),
        )
        try:
            with pytest.raises(FutureTimeout):
                _ = second_result.result(timeout=0.1)
        finally:
            release.set()
        assert first_result.result(timeout=5).status == "accepted"
        assert second_result.result(timeout=5).status == "rejected"
    assert second_planner.applied is False
    assert graph.get_children(goal_id) == []
    assert second.read_coordinator_snapshot(session_id, 0).proposals[1].status == "stale"
    other_graph.close()
    graph.close()


def test_other_connection_can_write_while_plan_is_prepared(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    entered, release = Event(), Event()
    delegate = _planner(graph, tmp_path)

    class SlowPlanner(_PreparedStub):
        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            return delegate.propose(goal, project_root, target_goal_id=target_goal_id)

        @override
        def prepare_proposal(self, proposal: PlanProposal, project_root: Path) -> BatchPlan:
            _ = (proposal, project_root)
            entered.set()
            assert release.wait(5)
            return BatchPlan((), (), "OPTIMAL")

        def apply_proposal(
            self,
            proposal: PlanProposal,
            *,
            target_goal_id: int,
            prepared_plan: BatchPlan,
        ) -> PlanResult:
            return delegate.apply_proposal(
                proposal, target_goal_id=target_goal_id, prepared_plan=prepared_plan
            )

    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(planner=cast(Planner, cast(object, SlowPlanner())))
    )
    session_id, goal_id = _start(control)
    _ = control.send_coordinator_command(session_id, PlanGoal("slow-plan"))
    other_graph = MikadoGraph(graph.db_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        approval = pool.submit(
            control.send_coordinator_command,
            session_id,
            DecidePlanProposal("approve-slow", "slow-plan", "accepted"),
        )
        assert entered.wait(5)
        try:
            _ = pool.submit(other_graph.add_node, "External", goal_id).result(timeout=2)
        finally:
            release.set()
        assert approval.result(timeout=5).status == "rejected"
    assert [node.description for node in graph.get_children(goal_id)] == ["External"]
    assert control.read_coordinator_snapshot(session_id, 0).proposals[0].status == "stale"
    other_graph.close()
    graph.close()


def test_other_connection_cannot_write_during_plan_apply(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    entered, release = Event(), Event()
    delegate = _planner(graph, tmp_path)

    class BlockingPlanner(_PreparedStub):
        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            return delegate.propose(goal, project_root, target_goal_id=target_goal_id)

        def apply_proposal(
            self,
            proposal: PlanProposal,
            *,
            target_goal_id: int,
            prepared_plan: BatchPlan,
        ) -> PlanResult:
            entered.set()
            assert release.wait(5)
            return delegate.apply_proposal(
                proposal, target_goal_id=target_goal_id, prepared_plan=prepared_plan
            )

    control = CoordinatorControl(
        graph,
        tmp_path,
        CoordinatorServices(planner=cast(Planner, cast(object, BlockingPlanner()))),
    )
    session_id, goal_id = _start(control)
    _ = control.send_coordinator_command(session_id, PlanGoal("blocked"))
    other_graph = MikadoGraph(graph.db_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        approval = pool.submit(
            control.send_coordinator_command,
            session_id,
            DecidePlanProposal("approve", "blocked", "accepted"),
        )
        assert entered.wait(5)
        external_write = pool.submit(other_graph.add_node, "External", goal_id)
        try:
            with pytest.raises(FutureTimeout):
                _ = external_write.result(timeout=0.1)
        finally:
            release.set()
        assert approval.result(timeout=5).status == "accepted"
        _ = external_write.result(timeout=5)
    assert [node.description for node in graph.get_children(goal_id)] == [
        "Implement task",
        "External",
    ]
    other_graph.close()
    graph.close()


def test_graph_revision_preserves_caller_transaction(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    conn = graph.group_connection
    initial = graph_revision(conn)
    _ = conn.execute("BEGIN")
    _ = conn.execute("UPDATE graph_revision SET revision = revision + 1 WHERE id = 1")
    assert graph_revision(conn) == initial + 1
    assert conn.in_transaction
    conn.rollback()
    assert graph_revision(conn) == initial
    graph.close()


def test_plan_graph_mutators_rollback_as_one_transaction(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    _, goal_id = _start(control)
    manifest = decode_manifest(
        {
            "manifest_version": "milknado.plan.v2",
            "goal": "Deliver",
            "goal_summary": "Deliver",
            "changes": [
                {"id": "a", "path": "src/a.py", "description": "First"},
                {"id": "b", "path": "src/b.py", "description": "Second", "depends_on": ["a"]},
            ],
            "new_relationships": [],
        }
    )
    plan = BatchPlan((Batch(0, ("a",), ()), Batch(1, ("b",), (0,))), (), "OPTIMAL")
    conn = graph.group_connection
    revision = cast(
        tuple[int], conn.execute("SELECT revision FROM graph_revision WHERE id = 1").fetchone()
    )[0]
    with pytest.raises(RuntimeError, match="abort"):
        with graph.plan_transaction(revision) as current:
            assert current
            assert len(apply_batches_to_graph(graph, plan, manifest, parent_id=goal_id)) == 2
            assert len(graph.get_children(goal_id)) > 0
            raise RuntimeError("abort")
    assert graph.get_children(goal_id) == []
    assert conn.execute("SELECT COUNT(*) FROM batch_plans").fetchone()[0] == 0
    graph.close()
