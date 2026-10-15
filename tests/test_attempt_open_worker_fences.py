"""A failure never lands while the attempt's node still owns an open worker."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from milknado.domains.common import NodeStatus, RunResult, WorkerIdentity, WorkerOwner
from milknado.domains.graph import (
    FinalizeTaskFailure,
    GroupWorkspace,
    MikadoGraph,
    StartTaskAttempt,
)

_NOW = "2026-01-01T00:00:00+00:00"
_FAILED = RunResult("failed", 1, False, _NOW, error="worker failed")


def _attempt(tmp_path: Path, limit: int) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner", limit))
    graph.runs.start("owner", node.id, "run.log", _NOW, None, attempt_id="attempt")
    return graph, node.id


def _open_worker(graph: MikadoGraph, node_id: int) -> None:
    graph.runs.record_worker(
        WorkerOwner("owner", 2**31 - 1, 123.5, "owner", node_id),
        WorkerIdentity("inv-1", 2**31 - 2, 2**31 - 2, 456.5),
    )


def _assert_unchanged(graph: MikadoGraph, node_id: int) -> None:
    node = graph.get_node(node_id)
    assert node is not None and node.status is NodeStatus.RUNNING and node.run_id == "owner"
    attempt = graph.task_attempts.get("attempt")
    assert attempt is not None and attempt.outcome == "active"
    assert graph.task_attempts.failed_count(node_id) == 0


def test_counted_failure_is_refused_while_a_worker_is_open(tmp_path: Path) -> None:
    graph, node_id = _attempt(tmp_path, 3)
    assert graph.task_attempts.confirm_execution("attempt", "owner", "owner")
    _open_worker(graph, node_id)

    request = FinalizeTaskFailure(node_id, "attempt", "owner", "owner", 3, _FAILED)
    disposition = graph.task_attempts.fail(request)

    assert disposition.status == "lost_fence"
    _assert_unchanged(graph, node_id)
    graph.close()


def test_startup_failure_is_refused_while_a_worker_is_open(tmp_path: Path) -> None:
    graph, node_id = _attempt(tmp_path, 3)
    assert graph.task_attempts.record_startup_failure("attempt", "owner", "owner")
    _open_worker(graph, node_id)

    request = FinalizeTaskFailure(node_id, "attempt", "owner", "owner", 3, _FAILED)
    disposition = graph.task_attempts.fail(request)

    assert disposition.status == "lost_fence"
    _assert_unchanged(graph, node_id)
    graph.close()


def test_group_failure_is_refused_when_its_task_already_has_a_result(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    group = graph.groups.create("graph", (node.id,), GroupWorkspace("wt", "branch", None))
    reservation = graph.groups.start_task(group.id, node.id, "owner")
    owner = reservation.attempt_id
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", owner, owner, 1))
    graph.runs.start(owner, node.id, "run.log", _NOW, None, attempt_id="attempt")
    assert graph.task_attempts.confirm_execution("attempt", owner, owner)
    with closing(sqlite3.connect(graph.db_path)) as raw:
        _ = raw.execute(
            "UPDATE execution_group_tasks SET status = 'failed' WHERE node_id = ?", (node.id,)
        )
        raw.commit()

    request = FinalizeTaskFailure(
        node.id, "attempt", owner, owner, 1, _FAILED, group_id=group.id, group_result="failed"
    )
    disposition = graph.task_attempts.fail(request)

    assert disposition.status == "lost_fence"
    current = graph.get_node(node.id)
    assert current is not None and current.status is NodeStatus.RUNNING
    assert graph.task_attempts.failed_count(node.id) == 0
    graph.close()
