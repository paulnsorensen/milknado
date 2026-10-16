"""Publication-edge tests for durable task-attempt fences and exception outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.dispatch._async_types import AsyncRunRequest
from milknado.domains.dispatch._sync_completion import finish_attempt_exception
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


def _started_graph(
    path: Path, *, limit: int = 2, attempt_id: str = "attempt"
) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, attempt_id, "owner", "owner", limit))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id=attempt_id)
    return graph, node.id


def _failure(node_id: int, *, status: str = "failed") -> FinalizeTaskFailure:
    result = RunResult(status, 1, False, _NOW, error="worker failed")
    return FinalizeTaskFailure(node_id, "attempt", "owner", "owner", result)


@dataclass(frozen=True)
class _SyncRequest:
    node_id: int
    max_node_attempts: int = 2
    timeout_seconds: int = 1


def _async_request(root: Path, node_id: int, limit: int = 2) -> AsyncRunRequest:
    return AsyncRunRequest(
        root, node_id, "brief", None, 1, "owner", "worker", root, max_node_attempts=limit
    )


@pytest.mark.parametrize(
    "case",
    [
        ("", "owner", 2, "identities must be nonempty"),
        ("attempt", "", 2, "identities must be nonempty"),
        ("attempt", "owner", 0, "must be positive"),
        ("attempt", "owner", True, "must be positive"),
    ],
)
def test_start_rejects_invalid_identity_and_budget(
    tmp_path: Path, case: tuple[str, str, int, str]
) -> None:
    attempt_id, owner, limit, message = case
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    with pytest.raises(ValueError, match=message):
        _ = graph.task_attempts.start(
            StartTaskAttempt(node.id, attempt_id, owner, max_node_attempts=limit)
        )
    assert graph.task_attempts.get("attempt") is None
    graph.close()


def test_start_is_idempotent_but_owner_and_budget_are_pinned(tmp_path: Path) -> None:
    graph, node_id = _started_graph(tmp_path / "graph.db")
    original = graph.task_attempts.get("attempt")
    assert (
        graph.task_attempts.start(StartTaskAttempt(node_id, "attempt", "owner", "owner", 2))
        == original
    )
    for request in (
        StartTaskAttempt(node_id, "attempt", "different", max_node_attempts=2),
        StartTaskAttempt(node_id, "attempt", "owner", max_node_attempts=3),
    ):
        with pytest.raises(ValueError, match="belongs to another owner"):
            _ = graph.task_attempts.start(request)
    with pytest.raises(ValueError, match="owner fence lost"):
        _ = graph.task_attempts.start(StartTaskAttempt(node_id, "other", "different"))
    graph.close()


def test_invocation_and_nonfailure_fences_preserve_final_outcome(tmp_path: Path) -> None:
    graph, node_id = _started_graph(tmp_path / "graph.db")
    with pytest.raises(ValueError, match="identity must be nonempty"):
        _ = graph.task_attempts.bind_invocation("attempt", "owner", "")
    with pytest.raises(ValueError, match="identity must be nonempty"):
        _ = graph.task_attempts.bind_worker_invocation(("attempt", "owner", "owner", ""))
    assert graph.task_attempts.bind_invocation("attempt", "owner", "owner")
    assert not graph.task_attempts.bind_invocation("missing", "owner", "other")
    with pytest.raises(ValueError, match="invalid nonfailure"):
        _ = graph.task_attempts.record_nonfailure("attempt", "owner", "unknown")
    assert not graph.task_attempts.record_nonfailure("attempt", "other", "cancelled")
    assert graph.task_attempts.record_nonfailure("attempt", "owner", "cancelled")
    assert graph.task_attempts.record_nonfailure("attempt", "owner", "cancelled")
    assert not graph.task_attempts.record_nonfailure("attempt", "owner", "done")
    assert not graph.task_attempts.bind_invocation("attempt", "owner", "other")
    assert graph.task_attempts.fail(_failure(node_id)).status == "lost_fence"
    assert graph.task_attempts.failed_count(node_id) == 0
    graph.close()


def test_failure_rejects_wrong_result_and_missing_run(tmp_path: Path) -> None:
    graph, node_id = _started_graph(tmp_path / "graph.db")
    assert graph.task_attempts.confirm_execution("attempt", "owner", "owner")
    with pytest.raises(ValueError, match="must be failed"):
        _ = graph.task_attempts.fail(_failure(node_id, status="done"))
    with pytest.raises(ValueError, match="does not belong"):
        _ = graph.task_attempts.fail(_failure(node_id + 1))
    graph.runs.finish("owner", RunResult("done", 0, False, _NOW))
    assert graph.task_attempts.fail(_failure(node_id)).status == "lost_fence"
    assert graph.task_attempts.failed_count(node_id) == 0
    graph.close()


def test_startup_failure_result_is_final_and_does_not_spend_budget(tmp_path: Path) -> None:
    graph, node_id = _started_graph(tmp_path / "graph.db")
    assert graph.task_attempts.record_startup_failure("attempt", "owner", "owner")
    first = graph.task_attempts.fail(_failure(node_id))
    assert first.status == "startup_failed"
    assert first.failed_count == 0
    assert graph.task_attempts.fail(_failure(node_id)) == first
    assert graph.task_attempts.failed_count(node_id) == 0
    node = graph.get_node(node_id)
    assert node is not None and node.status is NodeStatus.FAILED
    graph.close()


def test_lost_owner_rejects_failure_without_reclassifying_attempt(tmp_path: Path) -> None:
    graph, node_id = _started_graph(tmp_path / "graph.db")
    assert graph.task_attempts.confirm_execution("attempt", "owner", "owner")
    assert graph.replace_run_id(node_id, "owner", "next")
    result = graph.task_attempts.fail(_failure(node_id))
    assert result.status == "lost_fence"
    assert graph.task_attempts.failed_count(node_id) == 0
    attempt = graph.task_attempts.get("attempt")
    assert attempt is not None and attempt.outcome == "active"
    graph.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_exception_after_exec_counts_once(tmp_path: Path, asynchronous: bool) -> None:
    graph, node_id = _started_graph(tmp_path / "graph.db", attempt_id="owner")
    assert graph.task_attempts.confirm_execution("owner", "owner", "owner")
    if asynchronous:
        request = _async_request(tmp_path, node_id)
        finish_attempt_exception(graph, request, ("owner", True), RuntimeError("crashed"))
    else:
        finish_attempt_exception(
            graph, _SyncRequest(node_id), ("owner", True), RuntimeError("crashed")
        )
    assert graph.task_attempts.failed_count(node_id) == 1
    node = graph.get_node(node_id)
    assert node is not None and node.status is NodeStatus.PENDING
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "failed"
    graph.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unconfirmed_exception_holds_bounded_attempt(tmp_path: Path, asynchronous: bool) -> None:
    graph, node_id = _started_graph(tmp_path / "graph.db", attempt_id="owner")
    if asynchronous:
        request = _async_request(tmp_path, node_id)
        finish_attempt_exception(graph, request, ("owner", True), RuntimeError("no proof"))
    else:
        finish_attempt_exception(
            graph, _SyncRequest(node_id), ("owner", True), RuntimeError("no proof")
        )
    assert graph.task_attempts.failed_count(node_id) == 0
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "running"
    node = graph.get_node(node_id)
    assert node is not None and node.status is NodeStatus.RUNNING
    graph.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        (1, RuntimeError("wrapper failed"), "preserved"),
        (2, FileNotFoundError("target absent"), "startup_failed"),
    ],
)
def test_preexec_exception_keeps_default_or_startup_policy(
    tmp_path: Path, asynchronous: bool, case: tuple[int, Exception, str]
) -> None:
    limit, error, expected_outcome = case
    graph, node_id = _started_graph(tmp_path / "graph.db", limit=limit, attempt_id="owner")
    if asynchronous:
        request = _async_request(tmp_path, node_id, limit)
        finish_attempt_exception(graph, request, ("owner", True), error)
    else:
        finish_attempt_exception(graph, _SyncRequest(node_id, limit), ("owner", True), error)
    attempt = graph.task_attempts.get("owner")
    assert attempt is not None and attempt.outcome == expected_outcome
    assert graph.task_attempts.failed_count(node_id) == 0
    node = graph.get_node(node_id)
    assert node is not None and node.status is NodeStatus.FAILED
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "failed"
    graph.close()


def test_tmux_exception_without_exec_evidence_keeps_owner(tmp_path: Path) -> None:
    graph, node_id = _started_graph(tmp_path / "graph.db", attempt_id="owner")
    request = _async_request(tmp_path, node_id)
    finish_attempt_exception(graph, request, ("owner", False), OSError("pane lost"))
    assert graph.task_attempts.failed_count(node_id) == 0
    node = graph.get_node(node_id)
    assert node is not None and node.status is NodeStatus.RUNNING
    run = graph.runs.get("owner")
    assert run is not None and run["status"] == "running"
    graph.close()
