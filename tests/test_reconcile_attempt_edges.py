"""Recovery branches that refuse to guess when a task attempt is missing or already settled."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.adapters.process import ProcessAdapter
from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.dispatch.reconcile import fail_stale_running_runs, reconcile_node_status
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"
_FAILED = RunResult("failed", 1, False, _NOW, error="worker failed")


def _running_node(tmp_path: Path) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner", 3))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="attempt")
    return graph, node.id


def _lose_attempt_row(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "graph.db")) as raw:
        _ = raw.execute("DELETE FROM task_attempts WHERE attempt_id = 'attempt'")
        raw.commit()


def test_stale_tmux_recovery_rejects_missing_attempt(tmp_path: Path) -> None:
    graph, node_id = _running_node(tmp_path)
    _lose_attempt_row(tmp_path)
    graph.runs.set_pid("owner", 999999999)
    _ = graph.runs.deposit_message("owner", "tmux_exec_nonce", "nonce", _NOW)

    with pytest.raises(ValueError, match="attempt missing during tmux recovery"):
        _ = fail_stale_running_runs(graph, node_id, ProcessAdapter())
    graph.close()


def test_stale_recovery_rejects_missing_attempt(tmp_path: Path) -> None:
    graph, node_id = _running_node(tmp_path)
    _lose_attempt_row(tmp_path)
    graph.runs.set_pid("owner", 999999999)

    with pytest.raises(ValueError, match="attempt missing during stale recovery"):
        _ = fail_stale_running_runs(graph, node_id, ProcessAdapter())
    graph.close()


def test_node_reconcile_rejects_missing_attempt(tmp_path: Path) -> None:
    graph, node_id = _running_node(tmp_path)
    _lose_attempt_row(tmp_path)
    graph.runs.finish("owner", _FAILED)

    with pytest.raises(ValueError, match="attempt missing during recovery"):
        reconcile_node_status(graph, node_id, "failed", run_id="owner")
    graph.close()


def test_node_reconcile_rejects_cancellation_it_cannot_record(tmp_path: Path) -> None:
    graph, node_id = _running_node(tmp_path)
    graph.runs.finish("owner", RunResult("failed", -1, False, _NOW, error="cancelled"))
    _settle_and_reclaim(graph, node_id)

    with pytest.raises(RuntimeError, match="task outcome fence lost"):
        reconcile_node_status(graph, node_id, "failed", run_id="owner")
    node = graph.get_node(node_id)
    assert node is not None and node.run_id == "successor"
    graph.close()


def test_node_reconcile_fails_node_after_startup_failure(tmp_path: Path) -> None:
    graph, node_id = _running_node(tmp_path)
    assert graph.task_attempts.record_startup_failure("attempt", "owner", "owner")
    graph.runs.finish("owner", _FAILED)

    reconcile_node_status(graph, node_id, "failed", run_id="owner")

    node = graph.get_node(node_id)
    assert node is not None and node.status is NodeStatus.FAILED
    assert graph.task_attempts.failed_count(node_id) == 0
    graph.close()


def _settle_and_reclaim(graph: MikadoGraph, node_id: int) -> None:
    assert graph.task_attempts.confirm_execution("attempt", "owner", "owner")
    _ = graph.task_attempts.fail(
        FinalizeTaskFailure(node_id, "attempt", "owner", "owner", _FAILED)
    )
    assert graph.claim_node(node_id, "successor", now=_NOW)


def test_node_reconcile_leaves_successor_after_settled_attempt(tmp_path: Path) -> None:
    graph, node_id = _running_node(tmp_path)
    _settle_and_reclaim(graph, node_id)

    reconcile_node_status(graph, node_id, "failed", run_id="owner")

    node = graph.get_node(node_id)
    assert node is not None and node.run_id == "successor"
    assert node.status is NodeStatus.RUNNING
    assert graph.task_attempts.failed_count(node_id) == 1
    graph.close()
