from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.coordinator.plans import begin_plan, finish_plan
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import PlanResult


def test_foreign_and_unfinished_plan_retries_keep_receipt(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        workflow = CoordinatorWorkflow(graph, conn)
        owner = workflow.start_goal("Owner", "codex")
        foreign = workflow.start_goal("Foreign", "codex")
        assert begin_plan(conn, owner.id, "operation-1") == (True, None)
        with pytest.raises(ValueError, match="another coordinator"):
            _ = begin_plan(conn, foreign.id, "operation-1")
        with pytest.raises(ValueError, match="no durable result"):
            _ = begin_plan(conn, owner.id, "operation-1")
        assert conn.execute(
            "SELECT session_id, result_json FROM coordinator_plans WHERE operation_id = ?",
            ("operation-1",),
        ).fetchone() == (owner.id, None)
    graph.close()


def test_duplicate_plan_completion_preserves_first_result(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session = CoordinatorWorkflow(graph, conn).start_goal("Goal", "codex")
        assert begin_plan(conn, session.id, "operation-1") == (True, None)
        first = PlanResult(True, 0, tmp_path / "context.json")
        finish_plan(conn, "operation-1", first)
        with pytest.raises(ValueError, match="already recorded"):
            finish_plan(conn, "operation-1", PlanResult(False, 1))
        assert begin_plan(conn, session.id, "operation-1") == (False, first)
    graph.close()
