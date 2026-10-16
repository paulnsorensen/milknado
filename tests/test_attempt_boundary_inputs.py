"""Malformed handoffs and absent execution proof fail closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.adapters._tmux_exec import main
from milknado.adapters.process import ProcessAdapter
from milknado.domains.coordinator._task_attempt_workflow import (
    finish_task_attempt,
    link_task_run,
)
from milknado.domains.dispatch.reconcile import fail_stale_running_runs
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import MikadoGraph, TaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


def test_native_launch_rejects_disappeared_group(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    attempt = TaskAttempt("missing-group", node.id, "run", "attempt")
    with pytest.raises(ValueError, match="execution group disappeared"):
        link_task_run(graph, attempt, 2)
    assert graph.runs.get("attempt") is None
    graph.close()


def test_native_finish_requires_linked_attempt(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    attempt = TaskAttempt("missing-group", node.id, "run", "attempt")
    with pytest.raises(ValueError, match="native task attempt is not linked"):
        _ = finish_task_attempt(graph, attempt, NodeLoopOutcome(node.id, False), "running")
    assert graph.task_attempts.failed_count(node.id) == 0
    graph.close()


def test_tmux_wrapper_rejects_missing_arguments() -> None:
    assert main([]) == 125


def test_stale_run_without_timeout_cannot_be_inferred_dead(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    graph.runs.start("owner", node.id, "run.log", _NOW, None)
    assert fail_stale_running_runs(graph, node.id, ProcessAdapter()) == []
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "running"
    graph.close()
