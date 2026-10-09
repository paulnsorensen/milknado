from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.coordinator.planning_workflow import CoordinatorPlanning
from milknado.domains.coordinator.plans import get_proposal, transition_proposal
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import PlanChangeManifest, Planner, PlanProposal


class _PlannerStub:
    def __init__(self) -> None:
        self.calls = 0

    def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
        assert target_goal_id > 0
        self.calls += 1
        manifest = PlanChangeManifest("milknado.plan.v2", goal, goal, None, (), ())
        return PlanProposal(manifest, project_root / "context.md")


def test_proposal_replays_without_replanning_and_rejects_foreign_owner(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    planner = _PlannerStub()
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        owner = workflow.start_goal("Owner", "codex")
        foreign = workflow.start_goal("Foreign", "codex")
        planning = CoordinatorPlanning(graph, conn)
        first = planning.plan_goal(
            owner, cast(Planner, cast(object, planner)), tmp_path, "plan-1"
        )
        assert first.status == "pending"
        assert first.context_path == str(tmp_path / "context.md")
        assert planner.calls == 1
        assert planning.plan_goal(
            owner, cast(Planner, cast(object, planner)), tmp_path, "plan-1"
        ) == first
        with pytest.raises(ValueError, match="another coordinator"):
            planning.plan_goal(
                foreign, cast(Planner, cast(object, planner)), tmp_path, "plan-1"
            )
        with pytest.raises(ValueError, match="another coordinator"):
            planning.decide_plan(
                foreign, cast(Planner, cast(object, planner)), tmp_path, "plan-1", "rejected"
            )
        assert get_proposal(conn, "plan-1") == first
    with closing(sqlite3.connect(graph.db_path)) as reopened:
        assert get_proposal(reopened, "plan-1") == first
    assert planner.calls == 1
    graph.close()


def test_proposal_transition_is_write_once_and_preserves_manifest(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    planner = cast(Planner, cast(object, _PlannerStub()))
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session = CoordinatorWorkflow(graph, conn).start_goal("Goal", "codex")
        planning = CoordinatorPlanning(graph, conn)
        pending = planning.plan_goal(session, planner, tmp_path, "plan-1")
        rejected = transition_proposal(conn, pending.id, "pending", "rejected")
        assert rejected.status == "rejected"
        assert rejected.manifest == pending.manifest
        assert planning.decide_plan(session, planner, tmp_path, pending.id, "rejected") == rejected
        with pytest.raises(ValueError, match="already decided"):
            planning.decide_plan(session, planner, tmp_path, pending.id, "accepted")
        with pytest.raises(ValueError, match="state changed"):
            transition_proposal(conn, pending.id, "pending", "applied")
        assert get_proposal(conn, pending.id) == rejected
    graph.close()
