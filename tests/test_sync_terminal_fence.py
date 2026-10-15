"""A sync dispatch failure refuses to close a node that a sibling run still works."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from milknado.domains.common import WorkerIdentity, WorkerOwner
from milknado.domains.dispatch._sync_completion import finish_sync_exception
from milknado.domains.graph import MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class _SyncRequest:
    node_id: int
    max_node_attempts: int
    timeout_seconds: int = 1


def test_sibling_open_worker_blocks_the_terminal_node_write(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "owner", "owner", "owner", 1))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="owner")
    graph.runs.start("linked", node.id, "linked.log", _NOW, 1, attempt_id="owner")
    graph.runs.record_worker(
        WorkerOwner("linked", 2**31 - 1, 123.5, "linked", node.id),
        WorkerIdentity("inv-1", 2**31 - 2, 2**31 - 2, 456.5),
    )

    with pytest.raises(RuntimeError, match="terminal node fence lost for run owner"):
        finish_sync_exception(graph, _SyncRequest(node.id, 1), "owner", RuntimeError("x"))
    graph.close()
