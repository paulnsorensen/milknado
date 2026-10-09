from pathlib import Path
from typing import cast

import pytest

from milknado.domains.coordinator.plans import begin_plan, finish_plan
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import PlanResult


def test_plan_reservation_rejects_missing_identity_and_cross_owner(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    conn = graph.group_connection
    workflow = CoordinatorWorkflow(graph, conn)
    owner = workflow.start_goal("Owner", "codex")
    other = workflow.start_goal("Other", "codex")

    with pytest.raises(ValueError, match="identity"):
        _ = begin_plan(conn, owner.id, "")
    table = cast(
        tuple[str] | None,
        conn.execute("SELECT name FROM sqlite_master WHERE name = 'coordinator_plans'").fetchone(),
    )
    assert table is None

    assert begin_plan(conn, owner.id, "plan-1") == (True, None)
    with pytest.raises(ValueError, match="another coordinator"):
        _ = begin_plan(conn, other.id, "plan-1")
    assert conn.execute(
        "SELECT session_id, result_json FROM coordinator_plans WHERE operation_id = 'plan-1'"
    ).fetchone()[:] == (owner.id, None)
    graph.close()


def test_plan_completion_is_write_once_and_replays_result(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    conn = graph.group_connection
    session = CoordinatorWorkflow(graph, conn).start_goal("Deliver", "codex")
    assert begin_plan(conn, session.id, "plan-1") == (True, None)
    with pytest.raises(ValueError, match="no durable result"):
        _ = begin_plan(conn, session.id, "plan-1")

    result = PlanResult(True, 0, tmp_path / "context.md", nodes_created=2)
    finish_plan(conn, "plan-1", result)
    assert begin_plan(conn, session.id, "plan-1") == (False, result)
    stored = cast(
        tuple[str] | None,
        conn.execute(
            "SELECT result_json FROM coordinator_plans WHERE operation_id = 'plan-1'"
        ).fetchone(),
    )
    with pytest.raises(ValueError, match="already recorded"):
        finish_plan(conn, "plan-1", PlanResult(False, 1, None))
    unchanged = cast(
        tuple[str] | None,
        conn.execute(
            "SELECT result_json FROM coordinator_plans WHERE operation_id = 'plan-1'"
        ).fetchone(),
    )
    assert unchanged == stored
    graph.close()
