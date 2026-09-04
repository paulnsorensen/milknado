"""Focused recovery tests for persisted run ownership (#215, #309)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.dispatch import reconcile
from milknado.domains.graph import RunRecord


@dataclass
class _RecoveryNode:
    id: int
    status: NodeStatus
    run_id: str | None
    pid: int | None


class _RecoveryGraph:
    def __init__(self, *, pid: int | None, finish_succeeds: bool = True) -> None:
        self.node: _RecoveryNode = _RecoveryNode(
            id=1, status=NodeStatus.RUNNING, run_id="run-1", pid=pid
        )
        self.state: RunRecord = {
            "run_id": "run-1",
            "node_id": 1,
            "status": "running",
            "pid": None,
            "log_path": "",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": None,
            "timed_out": False,
            "exit_code": None,
            "error": None,
            "timeout_seconds": 300,
            "detail": None,
            "rebased": None,
        }
        self.finish_succeeds: bool = finish_succeeds

    def get_all_nodes(self) -> list[_RecoveryNode]:
        return [self.node]

    def get_node(self, node_id: int) -> _RecoveryNode | None:
        return self.node if node_id == self.node.id else None

    def runs_for_node(self, node_id: int, **_kwargs: object) -> list[RunRecord]:
        return [self.state] if node_id == self.node.id else []

    def finish_run(self, _run_id: str, result: RunResult) -> bool:
        if self.finish_succeeds:
            self.state.update(
                status=result.status,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                ended_at=result.ended_at,
                error=result.error,
            )
        return self.finish_succeeds

    def mark_terminal(self, _node_id: int, _run_id: str, status: NodeStatus) -> bool:
        self.node.status = status
        self.node.run_id = None
        return True


def _pid_dead(_pid: int) -> bool:
    return False


def _pid_alive(_pid: int) -> bool:
    return True


def test_reconcile_orphaned_runs_finalizes_dead_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _RecoveryGraph(pid=424242)
    monkeypatch.setattr(reconcile, "pid_alive", _pid_dead)

    recovered = reconcile.reconcile_orphaned_runs(graph)

    assert recovered == [graph.state]
    assert graph.state["error"] == "worker session gone"
    assert graph.node.status is NodeStatus.FAILED
    assert graph.node.run_id is None


def test_reconcile_orphaned_runs_preserves_healthy_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _RecoveryGraph(pid=424242)
    monkeypatch.setattr(reconcile, "pid_alive", _pid_alive)

    assert reconcile.reconcile_orphaned_runs(graph) == []
    assert graph.state["status"] == "running"
    assert graph.node.status is NodeStatus.RUNNING


def test_reconcile_orphaned_runs_ignores_graphs_without_node_enumeration() -> None:
    assert reconcile.reconcile_orphaned_runs(object()) == []


def test_dead_owner_recovery_rejects_lost_terminal_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _RecoveryGraph(pid=424242, finish_succeeds=False)
    monkeypatch.setattr(reconcile, "pid_alive", _pid_dead)

    with pytest.raises(RuntimeError, match="stale terminal write lost"):
        _ = reconcile.fail_stale_running_runs(graph, 1)
