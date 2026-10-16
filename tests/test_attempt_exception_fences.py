"""Exception paths preserve bounded execution-proof fences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from milknado.domains.common import NodeStatus
from milknado.domains.dispatch._async_types import AsyncRunRequest
from milknado.domains.dispatch._sync_completion import (
    SyncFailureRequest,
    confirm_execution,
    finish_attempt_exception,
)
from milknado.domains.graph import MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class _SyncRequest:
    node_id: int
    max_node_attempts: int
    timeout_seconds: int = 1


def _graph(path: Path, *, limit: int = 2, linked: bool = True) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    if linked:
        _ = graph.task_attempts.start(StartTaskAttempt(node.id, "owner", "owner", "owner", limit))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="owner" if linked else None)
    return graph, node.id


def _async_request(root: Path, node_id: int, limit: int = 2) -> AsyncRunRequest:
    return AsyncRunRequest(
        root, node_id, "brief", None, 1, "owner", "worker", root, max_node_attempts=limit
    )


def _finish_async(
    graph: MikadoGraph, request: AsyncRunRequest, exc: Exception, direct: bool
) -> None:
    finish_attempt_exception(graph, request, (request.run_id, direct), exc)


def _finish_sync(
    graph: MikadoGraph, request: SyncFailureRequest, run_id: str, exc: Exception
) -> None:
    finish_attempt_exception(graph, request, (run_id, True), exc)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_missing_attempt_is_not_silently_finished(tmp_path: Path, asynchronous: bool) -> None:
    graph, node_id = _graph(tmp_path / "graph.db", linked=False)
    with pytest.raises(ValueError, match="task attempt missing"):
        if asynchronous:
            _finish_async(graph, _async_request(tmp_path, node_id), RuntimeError("error"), True)
        else:
            _finish_sync(graph, _SyncRequest(node_id, 2), "owner", RuntimeError("error"))
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "running"
    graph.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_lost_owner_rejects_startup_evidence(tmp_path: Path, asynchronous: bool) -> None:
    graph, node_id = _graph(tmp_path / "graph.db")
    assert graph.replace_run_id(node_id, "owner", "next")
    with pytest.raises(RuntimeError, match="startup evidence fence lost"):
        if asynchronous:
            _finish_async(
                graph, _async_request(tmp_path, node_id), FileNotFoundError("target"), True
            )
        else:
            _finish_sync(graph, _SyncRequest(node_id, 2), "owner", FileNotFoundError("target"))
    assert graph.task_attempts.failed_count(node_id) == 0
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "running"
    graph.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_lost_owner_rejects_default_nonfailure(tmp_path: Path, asynchronous: bool) -> None:
    graph, node_id = _graph(tmp_path / "graph.db", limit=1)
    assert graph.replace_run_id(node_id, "owner", "next")
    with pytest.raises(RuntimeError, match="task outcome fence lost"):
        if asynchronous:
            _finish_async(
                graph, _async_request(tmp_path, node_id, 1), RuntimeError("wrapper"), True
            )
        else:
            _finish_sync(graph, _SyncRequest(node_id, 1), "owner", RuntimeError("wrapper"))
    node = graph.get_node(node_id)
    assert node is not None and node.status is NodeStatus.RUNNING
    graph.close()


def test_exec_confirmation_rejects_lost_owner(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path / "graph.db")
    assert graph.replace_run_id(node_id, "owner", "next")
    with pytest.raises(RuntimeError, match="proof fence lost"):
        confirm_execution(graph, "owner", 1234)
    run = graph.runs.get("owner")
    assert run is not None and run["pid"] == 1234
    attempt = graph.task_attempts.get("owner")
    assert attempt is not None and attempt.start_state == "unknown"
    graph.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_executed_exception_rejects_lost_owner(tmp_path: Path, asynchronous: bool) -> None:
    graph, node_id = _graph(tmp_path / "graph.db")
    assert graph.task_attempts.confirm_execution("owner", "owner", "owner")
    assert graph.replace_run_id(node_id, "owner", "next")
    with pytest.raises(RuntimeError, match="task attempt fence lost"):
        if asynchronous:
            _finish_async(graph, _async_request(tmp_path, node_id), RuntimeError("worker"), True)
        else:
            _finish_sync(graph, _SyncRequest(node_id, 2), "owner", RuntimeError("worker"))
    assert graph.task_attempts.failed_count(node_id) == 0
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "running"
    graph.close()
