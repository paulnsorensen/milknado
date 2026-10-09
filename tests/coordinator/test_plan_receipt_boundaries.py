from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import msgspec
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


def test_plan_result_receipt_keeps_all_fields_and_null_defaults(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session = CoordinatorWorkflow(graph, conn).start_goal("Goal", "codex")
        populated = PlanResult(True, 0, tmp_path / "context.json", 4, 3, 2, "solved", 1, 0)
        for operation_id, result in (("populated", populated), ("defaults", PlanResult(False, 7))):
            assert begin_plan(conn, session.id, operation_id) == (True, None)
            finish_plan(conn, operation_id, result)
            assert begin_plan(conn, session.id, operation_id) == (False, result)
        row = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT result_json FROM coordinator_plans WHERE operation_id = 'populated'"
            ).fetchone(),
        )
        assert row is not None
        assert json.loads(row[0]) == {
            "success": True,
            "exit_code": 0,
            "context_path": str(tmp_path / "context.json"),
            "nodes_created": 4,
            "batch_count": 3,
            "oversized_count": 2,
            "solver_status": "solved",
            "change_count": 1,
            "mega_batch_change_count": 0,
        }
        row = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT result_json FROM coordinator_plans WHERE operation_id = 'defaults'"
            ).fetchone(),
        )
        assert row is not None
        assert json.loads(row[0])["context_path"] is None
        assert json.loads(row[0])["mega_batch_change_count"] is None
    graph.close()


def test_plan_retry_rejects_malformed_saved_field(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session = CoordinatorWorkflow(graph, conn).start_goal("Goal", "codex")
        assert begin_plan(conn, session.id, "operation-1") == (True, None)
        finish_plan(conn, "operation-1", PlanResult(True, 0))
        with conn:
            _ = conn.execute(
                "UPDATE coordinator_plans SET result_json = "
                + "json_set(result_json, '$.success', 'yes') WHERE operation_id = 'operation-1'"
            )
        with pytest.raises(msgspec.ValidationError, match="success"):
            _ = begin_plan(conn, session.id, "operation-1")
    graph.close()
