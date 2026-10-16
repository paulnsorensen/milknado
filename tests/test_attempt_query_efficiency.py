"""Admission queries use indexes; repeated start evidence skips the second write."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from milknado.domains.graph import MikadoGraph, StartTaskAttempt
from milknado.domains.graph._task_attempt_binding import (
    _BIND_WORKER_SQL,  # pyright: ignore[reportPrivateUsage]
)
from milknado.domains.graph._task_attempts import record_start_evidence

_NOW = "2026-01-01T00:00:00+00:00"
_BIND_PARAMS = ("inv",) * 4 + ("owner",) * 9


def _graph(path: Path) -> MikadoGraph:
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "owner", "owner", "owner", 2))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="owner")
    return graph


def test_bind_worker_admission_plan_uses_new_indexes(tmp_path: Path) -> None:
    graph = _graph(tmp_path / "graph.db")
    rows = cast(
        list[tuple[int, int, int, str]],
        graph.group_connection.execute(
            "EXPLAIN QUERY PLAN " + _BIND_WORKER_SQL, _BIND_PARAMS
        ).fetchall(),
    )
    plan = "\n".join(row[3] for row in rows)
    assert "idx_runs_attempt_node" in plan
    assert "idx_run_workers_live_node" in plan
    assert "SCAN latest" not in plan and "SCAN live" not in plan
    graph.close()


def test_repeated_start_evidence_issues_no_second_update(tmp_path: Path) -> None:
    graph = _graph(tmp_path / "graph.db")
    conn = graph.group_connection
    identity = ("owner", "owner", "owner")
    assert record_start_evidence(conn, identity, "executed")
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    assert record_start_evidence(conn, identity, "executed")
    conn.set_trace_callback(None)
    assert not [s for s in statements if s.lstrip().upper().startswith("UPDATE")]
    graph.close()


def test_unchanged_start_evidence_still_enforces_fences(tmp_path: Path) -> None:
    graph = _graph(tmp_path / "graph.db")
    conn = graph.group_connection
    assert record_start_evidence(conn, ("owner", "owner", "owner"), "executed")
    assert not record_start_evidence(conn, ("owner", "stale", "owner"), "executed")
    assert not record_start_evidence(conn, ("owner", "owner", "other"), "executed")
    assert not record_start_evidence(conn, ("owner", "owner", "owner"), "released")
    attempt = graph.task_attempts.get("owner")
    assert attempt is not None and attempt.start_state == "executed"
    graph.close()
