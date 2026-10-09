from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.coordinator.control_models import (
    AttemptCommand,
    CreateGroup,
    DispatchTask,
    StartGoal,
    StartTurn,
)
from milknado.domains.coordinator.control_services import TurnRuntimeRequest, TurnRuntimeResult
from milknado.domains.graph import MikadoGraph


class _UnexpectedRuntime:
    def __init__(self) -> None:
        self.calls: int = 0

    def run(self, request: TurnRuntimeRequest) -> TurnRuntimeResult:
        _ = request
        self.calls += 1
        raise AssertionError("invalid turn reached the provider runtime")


def _launched(
    graph: MikadoGraph, root: Path, runtime: _UnexpectedRuntime
) -> tuple[CoordinatorControl, str, int, str, str]:
    control = CoordinatorControl(graph, root, CoordinatorServices(turn_runtime=runtime))
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session = cast(dict[str, object], started.result)
    session_id = cast(str, session["id"])
    task = graph.add_node("Implement", cast(int, session["goal_id"]))
    created = control.send_coordinator_command(
        session_id, CreateGroup("group", "main", (task.id,), str(root / "worktree"), "branch")
    )
    group_id = cast(str, cast(dict[str, object], created.result)["id"])
    dispatched = control.send_coordinator_command(
        session_id, DispatchTask("dispatch", group_id, task.id, "run")
    )
    attempt = cast(dict[str, object], cast(dict[str, object], dispatched.result)["attempt"])
    attempt_id = cast(str, attempt["attempt_id"])
    launched = control.send_coordinator_command(
        session_id, AttemptCommand("launch", group_id, task.id, "run", attempt_id)
    )
    assert launched.status == "accepted"
    return control, session_id, task.id, group_id, attempt_id


def test_empty_group_id_never_becomes_root_turn(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    runtime = _UnexpectedRuntime()
    control, session_id, task_id, group_id, attempt_id = _launched(graph, tmp_path, runtime)
    receipt = control.send_coordinator_command(
        session_id,
        StartTurn(
            "empty-group",
            "Work",
            group_id="",
            node_id=task_id,
            run_id="run",
            attempt_id=attempt_id,
        ),
    )
    assert receipt.status == "rejected"
    assert runtime.calls == 0
    group = graph.groups.get(group_id)
    assert group is not None and group.provider_session_id is None
    with sqlite3.connect(graph.db_path) as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM coordinator_turn_launches WHERE command_id = ?", ("empty-group",)
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM coordinator_provider_bindings WHERE coordinator_id = ?",
                (session_id,),
            ).fetchone()
            is None
        )
    graph.close()


def test_same_group_foreign_dispatch_cannot_launch_turn(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    runtime = _UnexpectedRuntime()
    control, session_id, task_id, group_id, attempt_id = _launched(graph, tmp_path, runtime)
    foreign = control.send_coordinator_command("", StartGoal("foreign", "Other", "codex"))
    foreign_id = cast(str, cast(dict[str, object], foreign.result)["id"])
    with sqlite3.connect(graph.db_path) as conn:
        _ = conn.execute(
            "UPDATE coordinator_dispatches SET session_id = ? WHERE attempt_id = ?",
            (foreign_id, attempt_id),
        )
    receipt = control.send_coordinator_command(
        session_id,
        StartTurn(
            "foreign-dispatch",
            "Work",
            group_id=group_id,
            node_id=task_id,
            run_id="run",
            attempt_id=attempt_id,
        ),
    )
    assert receipt.status == "rejected"
    assert runtime.calls == 0
    graph.close()
