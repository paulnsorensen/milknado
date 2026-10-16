"""Protected successor invocation admission uses durable worker cleanup."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import psutil
import pytest

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import NodeStatus, RunResult, WorkerOwner
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, WorkerProcess, spawn_gated
from milknado.loop._process_startup import _release_worker  # pyright: ignore[reportPrivateUsage]

_NOW = "2026-01-01T00:00:00+00:00"


def _context(path: Path, source: str, node_id: int) -> ProtectionContext:
    supervisor = psutil.Process()
    owner = WorkerOwner(source, supervisor.pid, supervisor.create_time(), source, node_id)
    return ProtectionContext(LoopWorkerEvidence(path), owner, path)


def _reap(worker: WorkerProcess) -> None:
    if worker.process.poll() is None:
        worker.process.kill()
    _ = worker.process.wait(timeout=3)


def _spawn(context: ProtectionContext, cleanup: ExitStack) -> WorkerProcess:
    worker = spawn_gated(
        SpawnOptions(
            command=(sys.executable, "-c", "pass"),
            cwd=None,
            env=None,
            text=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    )
    _ = cleanup.callback(_reap, worker)
    _ = cleanup.callback(worker.close_gate)
    context.evidence.record_worker(context.owner, worker.identity)
    return worker


def _seed(path: Path) -> tuple[MikadoGraph, int, ProtectionContext]:
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "source", 3))
    graph.runs.start("source", node.id, "source.log", _NOW, None, attempt_id="attempt")
    return graph, node.id, _context(path, "source", node.id)


@pytest.mark.skipif(os.name == "nt", reason="POSIX protected process gate")
def test_successor_waits_for_durable_predecessor_cleanup(tmp_path: Path) -> None:
    graph, node_id, context = _seed(tmp_path / "graph.db")
    with ExitStack() as cleanup:
        first = _spawn(context, cleanup)
        _release_worker(first, context)
        assert first.process.wait(timeout=3) == 0

        second = _spawn(context, cleanup)
        with pytest.raises(RuntimeError, match="invocation fence lost"):
            _release_worker(second, context)
        record = graph.task_attempts.get("attempt")
        assert record is not None
        assert record.active_invocation_id == first.identity.invocation_id
        context.evidence.end_worker(first.identity.invocation_id, 0)

        _release_worker(second, context)
        assert second.process.wait(timeout=3) == 0
        record = graph.task_attempts.get("attempt")
        assert record is not None
        assert record.active_invocation_id == second.identity.invocation_id
        assert record.start_state == "executed"
        assert graph.task_attempts.failed_count(node_id) == 0
        assert not graph.task_attempts.confirm_execution(
            "attempt", "owner", first.identity.invocation_id
        )
        context.evidence.end_worker(second.identity.invocation_id, 0)
    result = RunResult("failed", 1, False, _NOW, error="worker failed")
    disposition = graph.task_attempts.fail(
        FinalizeTaskFailure(node_id, "attempt", "owner", "source", result)
    )
    assert disposition.status is NodeStatus.PENDING
    assert disposition.failed_count == 1
    assert graph.task_attempts.failed_count(node_id) == 1
    graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX protected process gate")
def test_review_rotation_rejects_stale_source_after_successor_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph, node_id, old_context = _seed(path)
    graph.runs.start("new-source", node_id, "new.log", _NOW, None, attempt_id="attempt")
    assert graph.task_attempts.rotate_review_round("attempt", "owner", "new-owner", "new-source")
    current_context = _context(path, "new-source", node_id)
    with ExitStack() as cleanup:
        current = _spawn(current_context, cleanup)
        _release_worker(current, current_context)
        assert current.process.wait(timeout=3) == 0
        current_context.evidence.end_worker(current.identity.invocation_id, 0)

        delayed = _spawn(old_context, cleanup)
        with pytest.raises(RuntimeError, match="invocation fence lost"):
            _release_worker(delayed, old_context)
        record = graph.task_attempts.get("attempt")
        assert record is not None
        assert record.owner_run_id == "new-owner"
        assert record.active_invocation_id == current.identity.invocation_id
        assert record.start_state == "executed"
    graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX protected process gate")
def test_review_seed_refuses_delayed_old_source_before_current_worker(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph, node_id, old_context = _seed(path)
    graph.runs.start("new-owner", node_id, "new.log", _NOW, None, attempt_id="attempt")
    assert graph.task_attempts.rotate_review_round("attempt", "owner", "new-owner", "new-owner")
    with ExitStack() as cleanup:
        delayed = _spawn(old_context, cleanup)
        with pytest.raises(RuntimeError, match="invocation fence lost"):
            _release_worker(delayed, old_context)
        record = graph.task_attempts.get("attempt")
        assert record is not None
        assert record.active_invocation_id == "new-owner"
        assert record.start_state == "unknown"
        _reap(delayed)
        old_context.evidence.end_worker(delayed.identity.invocation_id, 0)
        current_context = _context(path, "new-owner", node_id)
        current = _spawn(current_context, cleanup)
        _release_worker(current, current_context)
        assert current.process.wait(timeout=3) == 0
        current_context.evidence.end_worker(current.identity.invocation_id, 0)
    graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX protected process gate")
@pytest.mark.parametrize("linked_owner", [False, True])
def test_owner_seed_accepts_latest_source_before_poststart_bind(
    tmp_path: Path, linked_owner: bool
) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "attempt", "owner", "owner"))
    if linked_owner:
        graph.runs.start("owner", node.id, "owner.log", _NOW, None, attempt_id="attempt")
    graph.runs.start("source", node.id, "source.log", _NOW, None, attempt_id="attempt")
    context = _context(path, "source", node.id)
    with ExitStack() as cleanup:
        worker = _spawn(context, cleanup)
        _release_worker(worker, context)
        assert worker.process.wait(timeout=3) == 0
        record = graph.task_attempts.get("attempt")
        assert record is not None
        assert record.active_invocation_id == worker.identity.invocation_id
        assert record.start_state == "executed"
        context.evidence.end_worker(worker.identity.invocation_id, 0)
    graph.close()
