"""Cancellation survives a partial commit, stale recovery, and a worker/cancel race."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.adapters.process import ProcessAdapter
from milknado.domains.common import MikadoNode, NodeStatus, RunResult
from milknado.domains.dispatch.reconcile import fail_stale_running_runs, reconcile_node_status
from milknado.domains.graph import MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


def _graph_with_attempt(tmp_path: Path) -> tuple[MikadoGraph, MikadoNode]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner", 3))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="attempt")
    return graph, node


def test_recovery_completes_partially_committed_cancellation(tmp_path: Path) -> None:
    graph, node = _graph_with_attempt(tmp_path)
    assert graph.task_attempts.record_nonfailure("attempt", "owner", "cancelled")

    flipped = fail_stale_running_runs(graph, node.id, ProcessAdapter())

    assert len(flipped) == 1
    assert flipped[0]["error"] == "cancelled"
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "failed" and run["error"] == "cancelled"
    reconcile_node_status(graph, node.id, "failed", run_id="owner")
    attempt = graph.task_attempts.get("attempt")
    assert attempt is not None and attempt.outcome == "cancelled"
    current = graph.get_node(node.id)
    assert current is not None and current.status is NodeStatus.FAILED
    assert graph.task_attempts.failed_count(node.id) == 0
    graph.close()


def _commit_worker_cancellation(graph: MikadoGraph, node_id: int) -> None:
    assert graph.task_attempts.record_nonfailure("attempt", "owner", "cancelled")
    graph.runs.finish("owner", RunResult("failed", -1, False, _NOW, error="cancelled"))
    assert graph.mark_terminal(node_id, "owner", NodeStatus.FAILED)


def _stale_first_read(graph: MikadoGraph, stale: MikadoNode, monkeypatch: pytest.MonkeyPatch):
    real = graph.get_node
    calls: list[int] = []

    def get_node(node_id: int) -> MikadoNode | None:
        calls.append(node_id)
        return stale if len(calls) == 1 else real(node_id)

    monkeypatch.setattr(graph, "get_node", get_node)


def test_cancel_accepts_worker_that_already_committed_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, node = _graph_with_attempt(tmp_path)
    stale = graph.get_node(node.id)
    assert stale is not None and stale.status is NodeStatus.RUNNING
    _commit_worker_cancellation(graph, node.id)
    _stale_first_read(graph, stale, monkeypatch)

    reconcile_node_status(graph, node.id, "failed", run_id="owner")

    current = graph.get_node(node.id)
    assert current is not None and current.status is NodeStatus.FAILED
    graph.close()


def test_cancel_still_fails_when_a_newer_owner_holds_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, node = _graph_with_attempt(tmp_path)
    stale = graph.get_node(node.id)
    assert stale is not None
    _commit_worker_cancellation(graph, node.id)
    assert graph.claim_node(node.id, "successor", now=_NOW)
    _stale_first_read(graph, stale, monkeypatch)

    with pytest.raises(RuntimeError, match="terminal node fence lost"):
        reconcile_node_status(graph, node.id, "failed", run_id="owner")

    current = graph.get_node(node.id)
    assert current is not None and current.run_id == "successor"
    assert current.status is NodeStatus.RUNNING
    graph.close()
