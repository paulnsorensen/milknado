"""Failure finalization notifies only for committed transitions."""

from __future__ import annotations

from pathlib import Path

from milknado.domains.common import MikadoNode, NodeStatus, PluginMeta, RunResult
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


class _Recorder:
    def __init__(self) -> None:
        self.events: list[NodeStatus] = []

    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(name="recorder", version="0.1.0", description="")

    def on_node_status_change(
        self, node: MikadoNode, old_status: NodeStatus, new_status: NodeStatus
    ) -> None:
        del node, old_status
        self.events.append(new_status)


def _failure(node_id: int, group_id: str | None) -> FinalizeTaskFailure:
    result = RunResult(status="failed", exit_code=1, timed_out=False, ended_at=_NOW, error="boom")
    return FinalizeTaskFailure(node_id, "attempt-1", "run-1", "run-1", result, group_id=group_id)


def _running_attempt(graph: MikadoGraph, recorder: _Recorder) -> int:
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "run-1", now=_NOW)
    _ = graph.task_attempts.start(
        StartTaskAttempt(node.id, "attempt-1", "run-1", max_node_attempts=3)
    )
    graph.runs.start("run-1", node.id, "run.log", _NOW, None, attempt_id="attempt-1")
    _ = graph.task_attempts.confirm_execution("attempt-1", "run-1", "run-1")
    recorder.events.clear()
    return node.id


def test_rolled_back_failure_publishes_no_notification(tmp_path: Path) -> None:
    recorder = _Recorder()
    graph = MikadoGraph(tmp_path / "graph.db", plugins=(recorder,))
    try:
        node_id = _running_attempt(graph, recorder)
        disposition = graph.task_attempts.fail(_failure(node_id, "missing-group"))
        assert disposition.status == "lost_fence"
        node = graph.get_node(node_id)
        assert node is not None
        assert node.status is NodeStatus.RUNNING
        assert graph.task_attempts.failed_count(node_id) == 0
        assert recorder.events == []
    finally:
        graph.close()


def test_committed_failure_publishes_exactly_one_notification(tmp_path: Path) -> None:
    recorder = _Recorder()
    graph = MikadoGraph(tmp_path / "graph.db", plugins=(recorder,))
    try:
        node_id = _running_attempt(graph, recorder)
        disposition = graph.task_attempts.fail(_failure(node_id, None))
        assert disposition.status is NodeStatus.PENDING
        assert recorder.events == [NodeStatus.PENDING]
    finally:
        graph.close()
