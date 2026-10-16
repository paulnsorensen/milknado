"""A dead owner does not release a task with unresolved start proof."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from milknado.adapters.process import ProcessAdapter
from milknado.domains.common import NodeStatus
from milknado.domains.dispatch.reconcile import fail_stale_running_runs
from milknado.domains.graph import MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


@pytest.mark.parametrize("state", ["unknown", "released"])
def test_dead_owner_unknown_start_keeps_retry_claim_fenced(tmp_path: Path, state: str) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    assert process.wait(timeout=3) == 0
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW, pid=process.pid)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "invocation", 3))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="attempt")
    if state == "released":
        assert graph.task_attempts.record_release("attempt", "owner", "invocation")
    graph.close()

    graph = MikadoGraph(path)
    assert fail_stale_running_runs(graph, node.id, ProcessAdapter()) == []
    assert not graph.try_reclaim(node.id, now=_NOW)
    assert not graph.claim_node(node.id, "next-owner", now=_NOW)
    current = graph.get_node(node.id)
    assert current is not None and current.status is NodeStatus.RUNNING
    assert current.run_id == "owner"
    assert graph.task_attempts.failed_count(node.id) == 0
    graph.close()


@pytest.mark.parametrize("limit", [None, 1])
def test_dead_owner_without_bounded_attempt_keeps_reclaim_path(
    tmp_path: Path, limit: int | None
) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    assert process.wait(timeout=3) == 0
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW, pid=process.pid)
    if limit is not None:
        _ = graph.task_attempts.start(
            StartTaskAttempt(node.id, "attempt", "owner", max_node_attempts=limit)
        )
    graph.close()

    graph = MikadoGraph(tmp_path / "graph.db")
    assert graph.try_reclaim(node.id, now=_NOW)
    assert graph.claim_node(node.id, "next-owner", now=_NOW)
    graph.close()


def test_confirmed_dead_execution_uses_attempt_transaction_before_reclaim(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    assert process.wait(timeout=3) == 0
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW, pid=process.pid)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "invocation", 3))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="attempt")
    assert graph.task_attempts.confirm_execution("attempt", "owner", "invocation")
    graph.close()

    graph = MikadoGraph(path)
    flipped = fail_stale_running_runs(graph, node.id, ProcessAdapter())
    assert len(flipped) == 1
    assert graph.task_attempts.failed_count(node.id) == 1
    assert not graph.try_reclaim(node.id, now=_NOW)
    assert graph.claim_node(node.id, "next-owner", now=_NOW)
    graph.close()


def test_confirmed_startup_failure_retains_default_terminal_policy(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    assert process.wait(timeout=3) == 0
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW, pid=process.pid)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "invocation", 3))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="attempt")
    assert graph.task_attempts.record_startup_failure("attempt", "owner", "invocation")
    graph.close()

    graph = MikadoGraph(path)
    assert len(fail_stale_running_runs(graph, node.id, ProcessAdapter())) == 1
    current = graph.get_node(node.id)
    assert current is not None and current.status is NodeStatus.FAILED
    assert graph.task_attempts.failed_count(node.id) == 0
    assert not graph.try_reclaim(node.id, now=_NOW)
    graph.close()
