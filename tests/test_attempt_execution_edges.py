"""Execution-slice attempt ownership and pre-start failure edges."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeStatus
from milknado.domains.execution._node_attempts import (
    AttemptRuntime,
    bind_before_start,
    bind_review_round,
    dispatch_identity,
    fail_bounded_attempt,
    record_nonfailure,
    supersede_review_round,
)
from milknado.domains.execution._node_context import ExecutionConfig, NodeExecutionContext
from milknado.domains.graph import MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


def _graph(path: Path) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "owner", "owner", "owner", 2))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="owner")
    return graph, node.id


def _context(root: Path, *, attempt_id: str = "owner") -> NodeExecutionContext:
    config = ExecutionConfig("codex", None, "pattern", root, max_node_attempts=2)
    return NodeExecutionContext(
        "owner", None, root, None, "main", "oid", None, 0, config, attempt_id=attempt_id
    )


def test_adopted_dispatch_reuses_parent_attempt_identity(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path / "graph.db")
    node = graph.get_node(node_id)
    assert node is not None
    run_id, attempt_id, selected_run_id, adopted = dispatch_identity(
        graph, node, _context(tmp_path).config, "owner"
    )
    assert run_id == "owner"
    assert attempt_id == "owner"
    assert selected_run_id is not None
    assert adopted
    graph.close()


def test_review_redispatch_rejects_lost_fence(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path / "graph.db")
    with pytest.raises(ValueError, match="review redispatch fence lost"):
        bind_review_round(graph, _context(tmp_path), "stale", "next")
    node = graph.get_node(node_id)
    assert node is not None and node.run_id == "owner"
    graph.close()


def test_record_nonfailure_cannot_replace_final_outcome(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path / "graph.db")
    assert graph.task_attempts.record_nonfailure("owner", "owner", "cancelled")
    node = graph.get_node(node_id)
    assert node is not None
    with pytest.raises(RuntimeError, match="outcome fence lost"):
        record_nonfailure(graph, node, _context(tmp_path), "done")
    attempt = graph.task_attempts.get("owner")
    assert attempt is not None and attempt.outcome == "cancelled"
    graph.close()


def test_bounded_failure_requires_persisted_attempt(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path / "graph.db")
    node = graph.get_node(node_id)
    assert node is not None
    runtime = AttemptRuntime(graph, lambda _node_id: None, lambda _node_id, _path: None)
    with pytest.raises(ValueError, match="task attempt missing"):
        fail_bounded_attempt(runtime, node, _context(tmp_path, attempt_id="missing"), "failed")
    assert graph.task_attempts.failed_count(node_id) == 0
    graph.close()


def test_prestart_callback_failure_finalizes_run(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path / "graph.db")

    def reject(_run_id: str) -> None:
        raise RuntimeError("admission rejected")

    with pytest.raises(RuntimeError, match="admission rejected"):
        bind_before_start(graph, "owner", reject)
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "failed"
    assert run["error"] == "dispatch aborted"
    graph.close()


def test_review_redispatch_needs_worker_run_identity(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    with pytest.raises(ValueError, match="no run fence/worker run id"):
        _ = supersede_review_round(lambda _id, _result: None, node, None)
    current = graph.get_node(node.id)
    assert current is not None and current.status is NodeStatus.PENDING
    graph.close()
