"""Operator reopen resets the attempt budget; failures record their attempt number."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from milknado.domains.common import NodeStatus, RunResult, WorkerIdentity
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt
from milknado.loop._process_gate import StartEvidence, WorkerProcess

_NOW = "2026-01-01T00:00:00+00:00"
_BUDGET = 3


def _fail_once(graph: MikadoGraph, node_id: int, name: str) -> NodeStatus:
    assert graph.claim_node(node_id, name, now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node_id, name, name, name, _BUDGET))
    graph.runs.start(name, node_id, f"{name}.log", _NOW, 1, attempt_id=name)
    assert graph.task_attempts.confirm_execution(name, name, name)
    result = RunResult("failed", 1, False, _NOW, error="boom", detail="worker detail")
    request = FinalizeTaskFailure(node_id, name, name, name, result)
    return NodeStatus(graph.task_attempts.fail(request).status)


def test_operator_reopen_resets_exhausted_budget(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert [_fail_once(graph, node.id, f"a{n}") for n in range(3)] == [
        NodeStatus.PENDING,
        NodeStatus.PENDING,
        NodeStatus.BLOCKED,
    ]
    assert graph.task_attempts.failed_count(node.id) == _BUDGET

    _ = graph.set_todo_status(node.id, NodeStatus.PENDING)

    assert graph.task_attempts.failed_count(node.id) == 0
    assert _fail_once(graph, node.id, "a4") is NodeStatus.PENDING
    graph.close()


def test_internal_requeue_keeps_failure_count(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert _fail_once(graph, node.id, "a1") is NodeStatus.PENDING
    assert graph.claim_node(node.id, "next", now=_NOW)
    assert graph.release(node.id, "next")
    assert graph.task_attempts.failed_count(node.id) == 1
    graph.close()


def test_counted_failure_records_attempt_number_on_run_row(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    _ = _fail_once(graph, node.id, "a1")
    _ = _fail_once(graph, node.id, "a2")
    graph.close()

    graph = MikadoGraph(path)
    run = graph.runs.get("a2")
    assert run is not None
    assert run["error"] == "attempt 2/3: boom"
    assert run["detail"] == "worker detail"
    graph.close()


def _running_attempt(graph: MikadoGraph, name: str) -> int:
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "provisional", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, name, "provisional", None, _BUDGET))
    graph.runs.start(name, node.id, f"{name}.log", _NOW, 1, attempt_id=name)
    return node.id


def test_adopt_owner_swaps_node_attempt_and_invocation_atomically(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node_id = _running_attempt(graph, "a1")

    assert graph.task_attempts.adopt_owner("a1", "provisional", "a1")

    node = graph.get_node(node_id)
    attempt = graph.task_attempts.get("a1")
    assert node is not None and node.run_id == "a1"
    assert attempt is not None
    assert (attempt.owner_run_id, attempt.active_invocation_id) == ("a1", "a1")
    graph.close()


def test_adopt_owner_miss_leaves_node_fence_unchanged(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node_id = _running_attempt(graph, "a1")

    assert not graph.task_attempts.adopt_owner("missing", "provisional", "a1")
    assert not graph.task_attempts.adopt_owner("a1", "wrong-owner", "a1")

    node = graph.get_node(node_id)
    assert node is not None and node.run_id == "provisional"
    graph.close()


def test_unconfirmed_failure_logs_attempt_identity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    assert graph.claim_node(graph.add_node("task").id, "a1", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(1, "a1", "a1", "a1", _BUDGET))
    graph.runs.start("a1", 1, "a1.log", _NOW, 1, attempt_id="a1")
    result = RunResult("failed", 1, False, _NOW, error="boom", detail="d")

    with caplog.at_level(logging.WARNING):
        disposition = graph.task_attempts.fail(FinalizeTaskFailure(1, "a1", "a1", "a1", result))

    assert disposition.status == "unknown_start"
    assert "node_id=1 run_id=a1 attempt_id=a1" in caplog.text
    graph.close()


@pytest.mark.parametrize("reported", [b"", b"X"])
def test_gate_that_does_not_report_ready_is_startup_failure(reported: bytes) -> None:
    gate_read, gate_write = os.pipe()
    status_read, status_write = os.pipe()
    process = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    _ = process.wait(timeout=30)
    worker = WorkerProcess(
        process, WorkerIdentity("inv", process.pid, process.pid, 0.0), gate_write, status_read
    )
    if reported:
        _ = os.write(status_write, reported)
    os.close(status_write)
    os.close(gate_read)
    try:
        worker.release()
        assert worker.start_evidence is StartEvidence.STARTUP_FAILED
    finally:
        worker.close_gate()
