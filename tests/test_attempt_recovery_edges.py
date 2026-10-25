"""Recovery tests for durable tmux execution receipts and missing attempts."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from threading import Thread
from typing import cast

import pytest

from milknado.domains.common import GitPort, MikadoNode, NodeStatus, RunResult
from milknado.domains.dispatch.cancel import cancel_run
from milknado.domains.dispatch.reconcile import fail_stale_running_runs
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, RunRecord, StartTaskAttempt
from milknado.loop._process_adapter import ProcessAdapter

pytestmark = pytest.mark.real_process

_NOW = "2026-01-01T00:00:00+00:00"


def _node(graph: MikadoGraph, node_id: int) -> MikadoNode:
    node = graph.get_node(node_id)
    assert node is not None
    return node


def _run(graph: MikadoGraph, run_id: str) -> RunRecord:
    run = graph.runs.get(run_id)
    assert run is not None
    return run


@pytest.mark.parametrize(
    ("receipt_state", "expected_status", "expected_failures"),
    [
        ("FINISHED", NodeStatus.PENDING, 1),
        ("EXEC_FAILED", NodeStatus.FAILED, 0),
    ],
)
def test_stale_tmux_receipt_sets_start_proof_before_failure(
    tmp_path: Path, receipt_state: str, expected_status: NodeStatus, expected_failures: int
) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    assert process.wait(timeout=3) == 0
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW, pid=process.pid)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner", 2))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="attempt")
    graph.runs.set_pid("owner", process.pid)
    nonce = "recovery-nonce"
    _ = graph.runs.deposit_message("owner", "tmux_exec_nonce", nonce, _NOW)
    receipt = tmp_path / "runs" / f"owner.{nonce}.exec.json"
    receipt.parent.mkdir()
    body: dict[str, object] = {"nonce": nonce, "state": receipt_state}
    if receipt_state == "FINISHED":
        body.update(pid=process.pid, returncode=1)
    else:
        body["error"] = "target missing"
    _ = receipt.write_text(json.dumps(body))
    graph.close()

    graph = MikadoGraph(tmp_path / "graph.db")
    flipped = fail_stale_running_runs(graph, node.id, ProcessAdapter())
    assert len(flipped) == 1
    assert graph.task_attempts.failed_count(node.id) == expected_failures
    current = graph.get_node(node.id)
    assert current is not None and current.status is expected_status
    attempt = graph.task_attempts.get("attempt")
    assert attempt is not None
    assert attempt.start_state == ("executed" if receipt_state == "FINISHED" else "startup_failed")
    assert fail_stale_running_runs(graph, node.id, ProcessAdapter()) == []
    graph.close()


def test_stale_run_with_bad_timestamp_does_not_spend_attempt(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner", 2))
    graph.runs.start("owner", node.id, "run.log", "invalid timestamp", 1, attempt_id="attempt")
    assert fail_stale_running_runs(graph, node.id, ProcessAdapter()) == []
    assert graph.task_attempts.failed_count(node.id) == 0
    current = graph.get_node(node.id)
    assert current is not None and current.status is NodeStatus.RUNNING
    graph.close()


def test_settled_attempt_closes_dead_linked_row_without_recount(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner", 2))
    graph.runs.start("owner", node.id, "owner.log", _NOW, 1, attempt_id="attempt")
    graph.runs.start("linked", node.id, "linked.log", _NOW, 1, attempt_id="attempt")
    assert graph.task_attempts.confirm_execution("attempt", "owner", "owner")
    disposition = graph.task_attempts.fail(
        FinalizeTaskFailure(
            node.id,
            "attempt",
            "owner",
            "owner",
            RunResult("failed", 1, False, _NOW, error="worker failed"),
        )
    )
    assert disposition.status is NodeStatus.PENDING
    assert graph.claim_node(node.id, "successor", now=_NOW)

    flipped = fail_stale_running_runs(graph, node.id, ProcessAdapter())
    assert len(flipped) == 1 and flipped[0]["run_id"] == "linked"
    linked = graph.runs.get("linked")
    assert linked is not None and linked["status"] == "failed"
    assert graph.task_attempts.failed_count(node.id) == 1
    current = graph.get_node(node.id)
    assert current is not None and current.run_id == "successor"
    assert current.status is NodeStatus.RUNNING
    assert fail_stale_running_runs(graph, node.id, ProcessAdapter()) == []
    graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cancellation")
@pytest.mark.parametrize("executed", [False, True])
def test_direct_pid_cancel_is_nonfailure_after_cleanup(tmp_path: Path, executed: bool) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    waiter = Thread(target=process.wait, daemon=True)
    waiter.start()
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node = graph.add_node("task")
        assert graph.claim_node(node.id, "owner", now=_NOW, pid=process.pid)
        _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner", 3))
        graph.runs.start("owner", node.id, "run.log", _NOW, 30, process.pid, attempt_id="attempt")
        if executed:
            assert graph.task_attempts.confirm_execution("attempt", "owner", "owner")
        result = cancel_run(graph, cast(GitPort, object()), ProcessAdapter(), tmp_path, "owner")
        assert result["error"] == "cancelled"
        assert result["status"] == "failed"
        attempt = graph.task_attempts.get("attempt")
        current = graph.get_node(node.id)
        assert attempt is not None and attempt.outcome == "cancelled"
        assert current is not None and current.status is NodeStatus.FAILED
        assert graph.task_attempts.failed_count(node.id) == 0
    finally:
        graph.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        waiter.join(timeout=5)
        assert not waiter.is_alive()


@pytest.mark.parametrize("confirmed", [False, True])
def test_stale_sweep_commits_linked_failure_only_with_exec_proof(
    tmp_path: Path, confirmed: bool
) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(
        StartTaskAttempt(node.id, "attempt", "owner", max_node_attempts=2)
    )
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="attempt")
    graph.runs.set_pid("owner", 999999999)
    if confirmed:
        assert graph.task_attempts.confirm_execution("attempt", "owner", "owner")
    else:
        assert graph.task_attempts.record_release("attempt", "owner", "owner")
    graph.close()

    graph = MikadoGraph(path)
    flipped = fail_stale_running_runs(graph, node.id, ProcessAdapter())
    assert len(flipped) == int(confirmed)
    assert graph.task_attempts.failed_count(node.id) == int(confirmed)
    assert _run(graph, "owner")["status"] == ("failed" if confirmed else "running")
    assert _node(graph, node.id).status is (
        NodeStatus.PENDING if confirmed else NodeStatus.RUNNING
    )
    graph.close()
