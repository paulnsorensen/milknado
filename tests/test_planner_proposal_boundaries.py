from __future__ import annotations

import json
from pathlib import Path
from typing import cast, final

import pytest

from milknado.domains.batching import BatchPlan
from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.common.protocols import CrgPort
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import Planner, PlanProposal, decode_manifest
from milknado.domains.planning.ports import PlanningPorts, PlanningProcessResult


@final
class _Process:
    def __init__(self, result: PlanningProcessResult) -> None:
        self.result = result
        self.validation = PlanningProcessResult(0)

    def run_agent(
        self, context_path: Path, command: str, project_root: Path
    ) -> PlanningProcessResult:
        _ = (context_path, command, project_root)
        return self.result

    def run_validation(
        self, command: str, payload: dict[str, object], project_root: Path
    ) -> PlanningProcessResult:
        _ = (command, payload, project_root)
        return self.validation


class _UnavailableCrg:
    def ensure_graph(self, project_root: Path) -> None:
        _ = project_root
        raise RuntimeError("CRG unavailable")


def _planner(graph: MikadoGraph, process: _Process) -> Planner:
    return Planner(
        graph,
        cast(CrgPort, cast(object, _UnavailableCrg())),
        "agent",
        PlanningPorts(process),
        "validate",
    )


def _manifest() -> dict[str, object]:
    return {
        "manifest_version": "milknado.plan.v2",
        "goal": "Deliver",
        "goal_summary": "Deliver",
        "changes": [],
        "new_relationships": [],
    }


def test_proposal_requires_existing_target_before_external_process(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    process = _Process(PlanningProcessResult(0, "unused"))

    with pytest.raises(ValueError, match="planning target does not exist"):
        _ = _planner(graph, process).propose("Deliver", tmp_path, target_goal_id=999_999)

    assert not (tmp_path / ".milknado" / "planning-context.md").exists()
    assert graph.get_all_nodes() == []
    graph.close()


def test_proposal_rejects_invalid_agent_output_without_graph_write(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    goal = graph.add_node("Deliver", spec=NodeSpec(kind=NodeKind.GOAL))
    process = _Process(PlanningProcessResult(0, "no manifest"))

    with pytest.raises(ValueError, match="valid proposal"):
        _ = _planner(graph, process).propose("Deliver", tmp_path, target_goal_id=goal.id)

    assert graph.get_children(goal.id) == []
    graph.close()


def test_proposal_rejects_external_validation_failure(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    goal = graph.add_node("Deliver", spec=NodeSpec(kind=NodeKind.GOAL))
    process = _Process(PlanningProcessResult(0, "```json\n" + json.dumps(_manifest()) + "\n```"))
    process.validation = PlanningProcessResult(1, stderr="policy refused")

    with pytest.raises(ValueError, match="proposal failed validation: policy refused"):
        _ = _planner(graph, process).propose("Deliver", tmp_path, target_goal_id=goal.id)

    assert graph.get_children(goal.id) == []
    graph.close()


def test_apply_proposal_refuses_task_target_without_graph_write(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    target = graph.add_node("Task")
    planner = _planner(graph, _Process(PlanningProcessResult(0)))
    proposal = PlanProposal(decode_manifest(_manifest()), tmp_path / "context.md")

    with pytest.raises(ValueError, match="planning target must be an existing goal"):
        _ = planner.apply_proposal(
            proposal,
            target_goal_id=target.id,
            prepared_plan=BatchPlan((), (), "OPTIMAL"),
        )

    assert graph.get_children(target.id) == []
    graph.close()
