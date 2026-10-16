"""Durable logical task-attempt accounting at the graph boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from milknado.domains.common import MikadoNode, NodeStatus, RunResult
from milknado.domains.dispatch.reconcile import reconcile_node_status
from milknado.domains.graph import (
    FinalizeTaskFailure,
    MikadoGraph,
    RunRecord,
    StartTaskAttempt,
    TaskAttemptRecord,
)

_NOW = "2026-01-01T00:00:00+00:00"


def _failed_result(error: str) -> RunResult:
    return RunResult(status="failed", exit_code=1, timed_out=False, ended_at=_NOW, error=error)


def _node(graph: MikadoGraph, node_id: int) -> MikadoNode:
    node = graph.get_node(node_id)
    assert node is not None
    return node


def _run(graph: MikadoGraph, run_id: str) -> RunRecord:
    run = graph.runs.get(run_id)
    assert run is not None
    return run


def _attempt(graph: MikadoGraph, attempt_id: str) -> TaskAttemptRecord:
    attempt = graph.task_attempts.get(attempt_id)
    assert attempt is not None
    return attempt


def test_failure_budget_and_run_linkage_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    for number in range(1, 4):
        owner = f"run-{number}"
        attempt_id = f"attempt-{number}"
        assert graph.claim_node(node.id, owner, now=_NOW)
        record = graph.task_attempts.start(
            StartTaskAttempt(node.id, attempt_id, owner, max_node_attempts=3)
        )
        assert record.attempt_id == attempt_id
        graph.runs.start(owner, node.id, "run.log", _NOW, None, attempt_id=attempt_id)
        _ = graph.task_attempts.confirm_execution(attempt_id, owner, owner)
        disposition = graph.task_attempts.fail(
            FinalizeTaskFailure(
                node.id, attempt_id, owner, owner, _failed_result(f"error-{number}")
            )
        )
        assert disposition.failed_count == number
        assert disposition.status is (NodeStatus.PENDING if number < 3 else NodeStatus.BLOCKED)
        assert f"error-{number}" in disposition.reason
        assert _run(graph, owner)["attempt_id"] == attempt_id
        assert (
            graph.task_attempts.fail(
                FinalizeTaskFailure(node.id, attempt_id, owner, owner, _failed_result("replay"))
            )
            == disposition
        )
        graph.close()
        graph = MikadoGraph(path)
        assert graph.task_attempts.failed_count(node.id) == number
    graph.close()


def test_attempt_history_survives_reset_but_cascades_on_delete_and_drop(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner"))
    assert graph.release(node.id, "owner")
    assert _attempt(graph, "attempt").node_id == node.id

    assert graph.delete_node(node.id) == 1
    assert graph.task_attempts.get("attempt") is None

    other = graph.add_node("other")
    assert graph.claim_node(other.id, "other-owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(other.id, "other-attempt", "other-owner"))
    assert graph.drop_all() == 1
    assert graph.task_attempts.get("other-attempt") is None
    graph.close()


def test_default_limit_keeps_failed_status(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner"))
    graph.runs.start("owner", node.id, "run.log", _NOW, None, attempt_id="attempt")
    _ = graph.task_attempts.confirm_execution("attempt", "owner", "owner")
    disposition = graph.task_attempts.fail(
        FinalizeTaskFailure(node.id, "attempt", "owner", "owner", _failed_result("failure"))
    )
    assert disposition.status is NodeStatus.FAILED
    assert disposition.failed_count == 1
    graph.close()


def test_unknown_start_holds_current_owner_without_spending_budget(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(
        StartTaskAttempt(node.id, "attempt", "owner", max_node_attempts=3)
    )
    graph.runs.start("owner", node.id, "run.log", _NOW, None, attempt_id="attempt")
    assert graph.task_attempts.record_release("attempt", "owner", "owner")

    disposition = graph.task_attempts.fail(
        FinalizeTaskFailure(node.id, "attempt", "owner", "owner", _failed_result("failure"))
    )

    assert disposition.status == "unknown_start"
    assert graph.task_attempts.failed_count(node.id) == 0
    assert _node(graph, node.id).status is NodeStatus.RUNNING
    assert _run(graph, "owner")["status"] == "running"
    graph.close()


def test_stale_invocation_evidence_cannot_resolve_unknown_start(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "old-owner", now=_NOW)
    _ = graph.task_attempts.start(
        StartTaskAttempt(node.id, "attempt", "old-owner", "old-invocation")
    )
    assert graph.replace_run_id(node.id, "old-owner", "new-owner")
    assert graph.task_attempts.bind_invocation("attempt", "new-owner", "new-invocation")

    assert not graph.task_attempts.confirm_execution("attempt", "old-owner", "old-invocation")
    assert _attempt(graph, "attempt").start_state == "unknown"
    assert graph.task_attempts.confirm_execution("attempt", "new-owner", "new-invocation")
    graph.close()


def test_owner_change_does_not_erase_same_invocation_start_proof(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "provisional", now=_NOW)
    _ = graph.task_attempts.start(
        StartTaskAttempt(node.id, "attempt", "provisional", "invocation")
    )
    assert graph.task_attempts.confirm_execution("attempt", "provisional", "invocation")
    assert graph.replace_run_id(node.id, "provisional", "worker")
    assert graph.task_attempts.bind_invocation("attempt", "worker", "invocation")
    assert _attempt(graph, "attempt").start_state == "executed"
    graph.close()


def test_stale_owner_failure_is_not_unknown_start(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "old-owner", now=_NOW)
    _ = graph.task_attempts.start(
        StartTaskAttempt(node.id, "attempt", "old-owner", max_node_attempts=3)
    )
    graph.runs.start("old-owner", node.id, "run.log", _NOW, None, attempt_id="attempt")
    assert graph.replace_run_id(node.id, "old-owner", "new-owner")
    disposition = graph.task_attempts.fail(
        FinalizeTaskFailure(node.id, "attempt", "old-owner", "old-owner", _failed_result("old"))
    )
    assert disposition.status == "lost_fence"
    assert graph.task_attempts.failed_count(node.id) == 0
    graph.close()


def test_review_round_rotation_changes_owner_and_invalidates_prior_proof(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("review")
    assert graph.claim_node(node.id, "old", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "old", "old"))
    assert graph.task_attempts.confirm_execution("attempt", "old", "old")
    graph.runs.start("new", node.id, "new.log", _NOW, None, attempt_id="attempt")

    assert graph.task_attempts.rotate_review_round("attempt", "old", "new", "new")

    assert _node(graph, node.id).run_id == "new"
    assert _attempt(graph, "attempt").start_state == "unknown"
    assert not graph.task_attempts.confirm_execution("attempt", "old", "old")
    assert graph.task_attempts.confirm_execution("attempt", "new", "new")
    graph.close()


@pytest.mark.parametrize("confirmed", [True, False])
def test_reopened_recovery_uses_pinned_attempt_limit(tmp_path: Path, confirmed: bool) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(
        StartTaskAttempt(node.id, "attempt", "owner", max_node_attempts=3)
    )
    graph.runs.start("owner", node.id, "run.log", _NOW, None, attempt_id="attempt")
    if confirmed:
        assert graph.task_attempts.confirm_execution("attempt", "owner", "owner")
    else:
        assert graph.task_attempts.record_release("attempt", "owner", "owner")
    graph.runs.finish("owner", _failed_result("recovered failure"))
    graph.close()

    graph = MikadoGraph(path)
    reconcile_node_status(graph, node.id, "failed", run_id="owner")
    expected = NodeStatus.PENDING if confirmed else NodeStatus.RUNNING
    assert _node(graph, node.id).status is expected
    assert graph.task_attempts.failed_count(node.id) == int(confirmed)
    graph.close()


@pytest.mark.parametrize("executed", [False, True])
def test_nonfailure_outcome_is_durable_and_not_reclassified(
    tmp_path: Path, executed: bool
) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(
        StartTaskAttempt(node.id, "attempt", "owner", max_node_attempts=3)
    )
    graph.runs.start("owner", node.id, "run.log", _NOW, None, attempt_id="attempt")
    if executed:
        assert graph.task_attempts.confirm_execution("attempt", "owner", "owner")
    graph.runs.finish("owner", _failed_result("cancelled"))
    graph.close()

    graph = MikadoGraph(path)
    reconcile_node_status(graph, node.id, "failed", run_id="owner")
    assert _node(graph, node.id).status is NodeStatus.FAILED
    assert _attempt(graph, "attempt").outcome == "cancelled"
    assert graph.task_attempts.failed_count(node.id) == 0
    reconcile_node_status(graph, node.id, "failed", run_id="owner")
    assert _attempt(graph, "attempt").outcome == "cancelled"
    assert graph.task_attempts.failed_count(node.id) == 0
    graph.close()


def test_two_connections_replay_one_failure_disposition(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    first = MikadoGraph(path)
    node = first.add_node("task")
    assert first.claim_node(node.id, "owner", now=_NOW)
    _ = first.task_attempts.start(
        StartTaskAttempt(node.id, "attempt", "owner", max_node_attempts=3)
    )
    first.runs.start("owner", node.id, "run.log", _NOW, None, attempt_id="attempt")
    assert first.task_attempts.confirm_execution("attempt", "owner", "owner")
    second = MikadoGraph(path)
    request = FinalizeTaskFailure(node.id, "attempt", "owner", "owner", _failed_result("race"))
    barrier = Barrier(3)

    def finish(graph: MikadoGraph) -> object:
        _ = barrier.wait()
        return graph.task_attempts.fail(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(finish, first)
        right = pool.submit(finish, second)
        _ = barrier.wait()
        assert left.result(timeout=5) == right.result(timeout=5)
    assert first.task_attempts.failed_count(node.id) == 1
    assert second.task_attempts.failed_count(node.id) == 1
    assert _node(first, node.id).status is NodeStatus.PENDING
    first.close()
    second.close()
