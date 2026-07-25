"""Focused recovery tests for persisted run ownership (#215, #309)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from milknado.domains.common import NodeStatus
from milknado.domains.dispatch import reconcile


class _RecoveryGraph:
    def __init__(self, *, pid: int | None, finish_succeeds: bool = True) -> None:
        self.node = SimpleNamespace(id=1, status=NodeStatus.RUNNING, run_id="run-1", pid=pid)
        self.state = {
            "run_id": "run-1",
            "status": "running",
            "node_id": 1,
            "pid": None,
            "started_at": "2026-01-01T00:00:00+00:00",
            "timeout_seconds": 300,
        }
        self.finish_succeeds = finish_succeeds

    def get_all_nodes(self) -> list[SimpleNamespace]:
        return [self.node]

    def get_node(self, node_id: int) -> SimpleNamespace | None:
        return self.node if node_id == self.node.id else None

    def runs_for_node(self, node_id: int, **_kwargs) -> list[dict]:
        return [self.state] if node_id == self.node.id else []

    def finish_run(self, _run_id: str, result) -> bool:  # noqa: ANN001
        if self.finish_succeeds:
            self.state.update(
                status=result.status,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                ended_at=result.ended_at,
                error=result.error,
            )
        return self.finish_succeeds

    def mark_terminal(self, _node_id: int, _run_id: str, status: NodeStatus) -> None:
        self.node.status = status
        self.node.run_id = None


def test_reconcile_orphaned_runs_finalizes_dead_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _RecoveryGraph(pid=424242)
    monkeypatch.setattr(reconcile, "pid_alive", lambda _pid: False)

    recovered = reconcile.reconcile_orphaned_runs(graph)

    assert recovered == [graph.state]
    assert graph.state["error"] == "worker session gone"
    assert graph.node.status is NodeStatus.FAILED
    assert graph.node.run_id is None


def test_reconcile_orphaned_runs_preserves_healthy_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _RecoveryGraph(pid=424242)
    monkeypatch.setattr(reconcile, "pid_alive", lambda _pid: True)

    assert reconcile.reconcile_orphaned_runs(graph) == []
    assert graph.state["status"] == "running"
    assert graph.node.status is NodeStatus.RUNNING


def test_reconcile_orphaned_runs_ignores_graphs_without_node_enumeration() -> None:
    assert reconcile.reconcile_orphaned_runs(object()) == []


def test_dead_owner_recovery_rejects_lost_terminal_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _RecoveryGraph(pid=424242, finish_succeeds=False)
    monkeypatch.setattr(reconcile, "pid_alive", lambda _pid: False)

    with pytest.raises(RuntimeError, match="stale terminal write lost"):
        reconcile.fail_stale_running_runs(graph, 1)
