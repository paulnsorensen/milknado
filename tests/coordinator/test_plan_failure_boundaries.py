from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.common.protocols import CrgPort
from milknado.domains.coordinator.model import CoordinatorSession
from milknado.domains.coordinator.persistence import (
    create_coordinator_tables,
    start_coordinator,
)
from milknado.domains.coordinator.planning_workflow import CoordinatorPlanning
from milknado.domains.coordinator.plans import (
    PlanProposalRecord,
    get_proposal,
    save_proposal,
    transition_proposal,
)
from milknado.domains.graph import MikadoGraph, graph_revision
from milknado.domains.planning import Planner, decode_manifest
from milknado.domains.planning.ports import PlanningPorts, PlanningProcessResult


class _NoPlanningProcess:
    def run_agent(
        self, context_path: Path, command: str, project_root: Path
    ) -> PlanningProcessResult:
        _ = (context_path, command, project_root)
        raise AssertionError("planning must not start")

    def run_validation(
        self, command: str, payload: dict[str, object], project_root: Path
    ) -> PlanningProcessResult:
        _ = (command, payload, project_root)
        raise AssertionError("validation must not start")


def _planner(graph: MikadoGraph) -> Planner:
    return Planner(graph, cast(CrgPort, MagicMock()), "agent", PlanningPorts(_NoPlanningProcess()))


def _planning_state(tmp_path: Path) -> tuple[MikadoGraph, CoordinatorPlanning, CoordinatorSession]:
    graph = MikadoGraph(tmp_path / "graph.db")
    goal = graph.add_node("Deliver", spec=NodeSpec(kind=NodeKind.GOAL))
    conn = graph.group_connection
    create_coordinator_tables(conn)
    session = start_coordinator(conn, goal.id, "codex")
    return graph, CoordinatorPlanning(graph, conn), session


def _proposal(graph: MikadoGraph, session: CoordinatorSession) -> PlanProposalRecord:
    manifest = decode_manifest(
        {
            "manifest_version": "milknado.plan.v2",
            "goal": "Deliver",
            "goal_summary": "Deliver",
            "changes": [],
            "new_relationships": [],
        }
    )
    return save_proposal(
        graph.group_connection,
        PlanProposalRecord(
            "proposal",
            session.id,
            manifest,
            "context.md",
            graph_revision(graph.group_connection),
            "pending",
        ),
    )


def test_foreign_proposal_cannot_be_read_or_decided(tmp_path: Path) -> None:
    graph, planning, session = _planning_state(tmp_path)
    original = _proposal(graph, session)
    foreign = CoordinatorSession("foreign", session.goal_id, "codex", session.created_at)
    planner = _planner(graph)

    with pytest.raises(ValueError, match="belongs to another coordinator"):
        _ = planning.plan_goal(foreign, planner, tmp_path, original.id)
    with pytest.raises(ValueError, match="belongs to another coordinator"):
        _ = planning.decide_plan(foreign, planner, tmp_path, original.id, "accepted")

    assert get_proposal(graph.group_connection, original.id) == original
    graph.close()


def test_missing_coordinator_goal_never_creates_proposal(tmp_path: Path) -> None:
    graph, planning, session = _planning_state(tmp_path)
    missing = CoordinatorSession(session.id, 999_999, session.provider, session.created_at)

    with pytest.raises(ValueError, match="coordinator goal does not exist"):
        _ = planning.plan_goal(missing, _planner(graph), tmp_path, "missing")

    with pytest.raises(KeyError, match="missing"):
        _ = get_proposal(graph.group_connection, "missing")
    graph.close()


@pytest.mark.parametrize(
    ("status", "decision", "message"),
    [
        ("stale", "accepted", "plan proposal is stale"),
        ("rejected", "accepted", "already decided"),
        ("applied", "rejected", "already decided"),
    ],
)
def test_decided_proposal_refuses_conflicting_decision(
    tmp_path: Path, status: str, decision: str, message: str
) -> None:
    graph, planning, session = _planning_state(tmp_path)
    original = _proposal(graph, session)
    stored = transition_proposal(graph.group_connection, original.id, "pending", status)
    planner = _planner(graph)

    with pytest.raises(ValueError, match=message):
        _ = planning.decide_plan(session, planner, tmp_path, original.id, decision)

    assert get_proposal(graph.group_connection, original.id) == stored
    graph.close()


def test_empty_plan_identity_does_not_write_history(tmp_path: Path) -> None:
    graph, planning, session = _planning_state(tmp_path)
    with pytest.raises(ValueError, match="plan identity must not be empty"):
        planning.record_plan(session, "", "accepted")
    assert (
        graph.group_connection.execute(
            "SELECT COUNT(*) FROM coordinator_links WHERE session_id = ?", (session.id,)
        ).fetchone()[0]
        == 0
    )
    graph.close()


def test_missing_graph_revision_raises_without_claiming_transaction(tmp_path: Path) -> None:
    graph, _, _ = _planning_state(tmp_path)
    conn = graph.group_connection
    with conn:
        _ = conn.execute("DELETE FROM graph_revision WHERE id = 1")

    with pytest.raises(RuntimeError, match="graph revision is unavailable"):
        _ = graph_revision(conn)

    assert conn.execute("SELECT COUNT(*) FROM graph_revision").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 1
    graph.close()
