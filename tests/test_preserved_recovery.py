"""Reopen recovery completes a preserved attempt like the live terminal write would."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.dispatch import reconcile
from milknado.domains.dispatch.reconcile import reconcile_orphaned_runs
from milknado.domains.graph import MikadoGraph, StartTaskAttempt
from milknado.loop._process_adapter import ProcessAdapter

_NOW = "2026-01-01T00:00:00+00:00"
_DEAD_PID = 999999


def _crashed_graph(tmp_path: Path, budget: int, *, preserved_written: bool) -> MikadoGraph:
    graph = MikadoGraph(tmp_path / "g.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "r1", now=_NOW, pid=_DEAD_PID)
    graph.set_worktree(node.id, "r1", "/tmp/wt", "milknado/1-x")
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "r1", "r1", "r1", budget))
    graph.runs.start("r1", node.id, "l", _NOW, 60, attempt_id="r1")
    graph.runs.finish("r1", RunResult("failed", 0, False, _NOW, rebased=False))
    if preserved_written:
        assert graph.task_attempts.record_nonfailure("r1", "r1", "preserved")
    return graph


@pytest.mark.parametrize("budget", [1, 3])
@pytest.mark.parametrize("preserved_written", [True, False])
def test_recovery_completes_preserved_attempt(
    tmp_path: Path, budget: int, preserved_written: bool
) -> None:
    graph = _crashed_graph(tmp_path, budget, preserved_written=preserved_written)

    reconciled = reconcile_orphaned_runs(graph, ProcessAdapter())

    node = graph.get_node(1)
    assert node is not None and node.status is NodeStatus.FAILED
    assert node.run_id is None
    assert node.worktree_path == "/tmp/wt" and node.branch_name == "milknado/1-x"
    assert graph.task_attempts.failed_count(1) == 0
    attempt = graph.task_attempts.get("r1")
    assert attempt is not None and attempt.outcome == "preserved"
    assert [run["run_id"] for run in reconciled] == ["r1"]
    graph.close()


def test_recovery_does_not_report_unreconciled_run(tmp_path: Path) -> None:
    graph = _crashed_graph(tmp_path, 3, preserved_written=True)
    assert graph.task_attempts.get("r1") is not None
    assert graph.mark_terminal(1, "r1", NodeStatus.FAILED, preserve_recovery=True)

    assert reconcile_orphaned_runs(graph, ProcessAdapter()) == []
    node = graph.get_node(1)
    assert node is not None and node.status is NodeStatus.FAILED
    graph.close()


def test_one_failing_node_rolls_back_finishes_sweep_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db_path = tmp_path / "g.db"
    graph = MikadoGraph(db_path)
    for name, run_id in (("broken", "r1"), ("healthy", "r2")):
        node = graph.add_node(name)
        assert graph.claim_node(node.id, run_id, now=_NOW, pid=_DEAD_PID)
        _ = graph.task_attempts.start(StartTaskAttempt(node.id, run_id, run_id, run_id, 1))
        graph.runs.start(run_id, node.id, "l", _NOW, 60, attempt_id=run_id)
        graph.runs.finish(run_id, RunResult("failed", 0, False, _NOW, rebased=False))
    settle = reconcile.reconcile_node_status

    def settle_unless_broken(
        graph: MikadoGraph, node_id: int, status: str, *, run_id: str
    ) -> None:
        if node_id == 1:
            _ = graph.group_connection.execute(
                "UPDATE nodes SET description = 'dirty' WHERE id = 1"
            )
            raise RuntimeError("injected settle failure")
        settle(graph, node_id, status, run_id=run_id)

    monkeypatch.setattr(reconcile, "reconcile_node_status", settle_unless_broken)

    with pytest.raises(ExceptionGroup) as raised:
        _ = reconcile_orphaned_runs(graph, ProcessAdapter())

    assert [type(exc) for exc in raised.value.exceptions] == [RuntimeError]
    assert "reconcile failed for node 1" in caplog.text
    graph.close()
    reopened = MikadoGraph(db_path)
    broken = reopened.get_node(1)
    assert broken is not None and broken.status is NodeStatus.RUNNING
    assert broken.description != "dirty"
    healthy = reopened.get_node(2)
    assert healthy is not None and healthy.status is NodeStatus.FAILED
    reopened.close()


def test_recovered_counted_failure_prefixes_error_on_already_failed_run(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "g.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "r1", now=_NOW, pid=_DEAD_PID)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "r1", "r1", "r1", 3))
    graph.runs.start("r1", node.id, "l", _NOW, 60, attempt_id="r1")
    assert graph.task_attempts.confirm_execution("r1", "r1", "r1")
    graph.runs.finish("r1", RunResult("failed", 1, False, _NOW, error="boom"))

    _ = reconcile_orphaned_runs(graph, ProcessAdapter())

    run = graph.runs.get("r1")
    assert run is not None and run["error"] == "attempt 1/3: boom"
    graph.close()
