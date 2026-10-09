from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import cast, final

from milknado.domains.common.protocols import CrgPort
from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import PlanGoal, StartGoal
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import Planner
from milknado.domains.planning.ports import PlanningPorts, PlanningProcessResult


class _UnavailableCrg:
    def ensure_graph(self, project_root: Path) -> None:
        _ = project_root
        raise RuntimeError("CRG is unavailable")


@final
class _WaitingProcess:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.calls = 0

    def run_agent(
        self, context_path: Path, command: str, project_root: Path
    ) -> PlanningProcessResult:
        _ = (context_path, command, project_root)
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("planner did not resume")
        return PlanningProcessResult(0)

    def run_validation(
        self, command: str, payload: dict[str, object], project_root: Path
    ) -> PlanningProcessResult:
        _ = (command, payload, project_root)
        raise AssertionError("no manifest requires no validation")


def test_snapshot_completes_while_planner_waits_and_retry_runs_once(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    process = _WaitingProcess()
    crg = cast(CrgPort, cast(object, _UnavailableCrg()))
    planner = Planner(graph, crg, "codex", PlanningPorts(process))
    control = CoordinatorControl(graph, tmp_path, CoordinatorServices(planner=planner))
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    command = PlanGoal("plan")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(control.send_coordinator_command, session_id, command)
            assert process.started.wait(timeout=2)
            try:
                with sqlite3.connect(graph.db_path) as conn:
                    receipt = cast(
                        tuple[str] | None,
                        conn.execute(
                            "SELECT status FROM coordinator_web_receipts WHERE command_id = ?",
                            (command.command_id,),
                        ).fetchone(),
                    )
                    reservation = cast(
                        tuple[str | None] | None,
                        conn.execute(
                            "SELECT result_json FROM coordinator_plans WHERE operation_id = ?",
                            (command.command_id,),
                        ).fetchone(),
                    )
                assert receipt == ("unconfirmed",)
                assert reservation == (None,)
                snapshot = pool.submit(control.read_coordinator_snapshot, session_id, 0)
                assert snapshot.result(timeout=1).goal.description == "Deliver"
                retry = control.send_coordinator_command(session_id, command)
                assert retry.status == "unconfirmed"
                assert process.calls == 1
            finally:
                process.release.set()
            completed = first.result(timeout=3)
        assert completed.status == "accepted"
        assert control.send_coordinator_command(session_id, command) == completed
        assert process.calls == 1
        snapshot = control.read_coordinator_snapshot(session_id, 0)
        assert [(event.kind, event.status) for event in snapshot.events][-2:] == [
            ("planning_decision", "accepted"),
            ("command", "accepted"),
        ]
    finally:
        process.release.set()
        graph.close()
