from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from threading import Event
from typing import cast, final

import pytest

from milknado.domains.common import MikadoNode, NodeSpec
from milknado.domains.common.protocols import CrgPort
from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import StartGoal
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.coordinator.persistence import get_coordinator
from milknado.domains.coordinator.planning_workflow import CoordinatorPlanning
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import Planner
from milknado.domains.planning.ports import PlanningPorts, PlanningProcessResult


class _UnavailableCrg:
    def ensure_graph(self, project_root: Path) -> None:
        _ = project_root
        raise RuntimeError("CRG is unavailable")


@final
class _PlanningProcess:
    def __init__(self) -> None:
        self.calls = 0
        self.second_started = Event()
        self.release_second = Event()

    def run_agent(
        self, context_path: Path, command: str, project_root: Path
    ) -> PlanningProcessResult:
        _ = (context_path, command, project_root)
        self.calls += 1
        if self.calls == 2:
            self.second_started.set()
            if not self.release_second.wait(timeout=5):
                raise TimeoutError("second proposal did not resume")
        manifest = {
            "manifest_version": "milknado.plan.v2",
            "goal": "Deliver",
            "goal_summary": "Deliver",
            "changes": [{"id": "task-1", "path": "src/a.py", "description": "Implement task"}],
            "new_relationships": [],
        }
        return PlanningProcessResult(0, f"```json\n{json.dumps(manifest)}\n```")

    def run_validation(
        self, command: str, payload: dict[str, object], project_root: Path
    ) -> PlanningProcessResult:
        _ = (command, payload, project_root)
        return PlanningProcessResult(0)


def _hold_graph_write(graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch) -> tuple[Event, Event]:
    node_written = Event()
    release_approval = Event()
    original_add_node = graph.add_node

    def add_node_then_wait(
        description: str,
        parent_id: int | None = None,
        spec: NodeSpec | None = None,
        files: tuple[str, ...] | None = None,
    ) -> MikadoNode:
        node = original_add_node(description, parent_id, spec, files)
        node_written.set()
        if not release_approval.wait(timeout=5):
            raise TimeoutError("approval did not resume")
        return node

    monkeypatch.setattr(graph, "add_node", add_node_then_wait)
    return node_written, release_approval


def _assert_durable(tmp_path: Path, session_id: str, goal_id: int, first_id: str) -> None:
    reopened = MikadoGraph(tmp_path / "graph.db")
    try:
        assert len(reopened.get_children(goal_id)) == 1
        snapshot = CoordinatorControl(reopened, tmp_path).read_coordinator_snapshot(session_id, 0)
        assert snapshot.proposals[0].status == "applied"
        decisions = [
            (event.entity_id, event.status)
            for event in snapshot.events
            if event.kind == "planning_decision"
        ]
        assert decisions == [(first_id, "accepted")]
    finally:
        reopened.close()


def test_shared_connection_keeps_approval_graph_write_when_second_plan_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    process = _PlanningProcess()
    crg = cast(CrgPort, cast(object, _UnavailableCrg()))
    planner = Planner(graph, crg, "codex", PlanningPorts(process))
    control = CoordinatorControl(graph, tmp_path, CoordinatorServices(planner=planner))
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    session = get_coordinator(graph.group_connection, session_id)
    assert session is not None
    workflow = CoordinatorPlanning(graph, graph.group_connection)
    first = workflow.plan_goal(session, planner, tmp_path, "first")
    node_written, release_approval = _hold_graph_write(graph, monkeypatch)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            second = pool.submit(workflow.plan_goal, session, planner, tmp_path, "second")
            assert process.second_started.wait(timeout=2)
            approval = pool.submit(
                workflow.decide_plan, session, planner, tmp_path, first.id, "accepted"
            )
            assert node_written.wait(timeout=2)
            process.release_second.set()
            try:
                _ = second.result(timeout=0.2)
            except (FutureTimeout, sqlite3.OperationalError):
                pass
            finally:
                release_approval.set()
            assert approval.result(timeout=3).status == "applied"
            assert len(graph.get_children(session.goal_id)) == 1
            with pytest.raises(ValueError, match="graph changed during planning"):
                _ = second.result(timeout=3)
    finally:
        process.release_second.set()
        release_approval.set()
        graph.close()

    _assert_durable(tmp_path, session_id, session.goal_id, first.id)
