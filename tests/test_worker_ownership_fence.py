from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import psutil
import pytest

from milknado.adapters import FlockSlotPool
from milknado.adapters.loop import LoopAdapter
from milknado.domains.common import (
    CrgPort,
    GitPort,
    LoopPort,
    NodeStatus,
    ObservationKey,
    WorkerIdentity,
    WorkerOwner,
)
from milknado.domains.common.errors import InvalidTransition
from milknado.domains.execution import Executor, PreservedWorkerRun
from milknado.domains.execution._stop import stop_graph_run
from milknado.domains.graph import HostCapacityFull, MikadoGraph
from milknado.loop import EventType, QueueEmitter, RunConfig
from milknado.loop._events import Event, NoData


@pytest.fixture
def worker() -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        text=True,
    )
    try:
        yield process
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        _ = process.wait(timeout=3)


def _owned_graph(
    tmp_path: Path, worker: subprocess.Popen[str], *, blocked: bool = False
) -> tuple[MikadoGraph, int, Path]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("owned")
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    assert graph.claim_node(node.id, "owner", now="2026-01-01T00:00:00+00:00")
    graph.set_worktree(node.id, "owner", str(worktree), "owned-branch")
    graph.runs.start("owner", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    if blocked:
        assert graph.mark_blocked_fenced(node.id, "owner")
    token = psutil.Process(worker.pid).create_time()
    graph.runs.record_worker(
        WorkerOwner("reviewer", os.getpid(), psutil.Process().create_time(), "owner", node.id),
        WorkerIdentity("invocation", worker.pid, worker.pid, token),
    )
    graph.runs.begin_worker_observation(
        ObservationKey("invocation", "supervisor", 1, -1, worker.pid, token)
    )
    return graph, node.id, worktree


def test_unconfirmed_worker_blocks_owner_release_and_terminal_transitions(
    tmp_path: Path, worker: subprocess.Popen[str]
) -> None:
    graph, node_id, worktree = _owned_graph(tmp_path, worker)
    try:
        assert graph.release(node_id, "owner") is False
        assert graph.mark_terminal(node_id, "owner", NodeStatus.FAILED) is False
        assert graph.mark_terminal(node_id, "owner", NodeStatus.DONE) is False
        assert graph.mark_blocked_fenced(node_id, "owner") is False
        node = graph.get_node(node_id)
        assert node is not None
        assert node.status is NodeStatus.RUNNING
        assert node.run_id == "owner"
        assert node.worktree_path == str(worktree)
        assert worktree.exists()
        assert graph.runs.live_workers(node_id=node_id)
    finally:
        graph.close()


def test_unconfirmed_worker_blocks_plain_failure_and_claim(
    tmp_path: Path, worker: subprocess.Popen[str]
) -> None:
    graph, node_id, worktree = _owned_graph(tmp_path, worker)
    try:
        with pytest.raises(InvalidTransition):
            graph.mark_failed(node_id)
        node = graph.get_node(node_id)
        assert node is not None and node.run_id == "owner"
        assert node.worktree_path == str(worktree)
        assert graph.claim_node(node_id, "new-owner", now="2026-01-01T00:01:00+00:00") is False
    finally:
        graph.close()


def test_terminal_thread_exit_does_not_confirm_ordinary_worker(
    tmp_path: Path, worker: subprocess.Popen[str]
) -> None:
    graph, node_id, worktree = _owned_graph(tmp_path, worker)
    try:
        adapter = LoopAdapter(graph=graph)
        with patch.object(adapter._manager, "stop_and_join", return_value=True):  # pyright: ignore[reportPrivateUsage]
            unconfirmed: set[str] = set()
            assert not stop_graph_run(adapter, unconfirmed, "owner", 0.1)
        assert unconfirmed == {"owner"}
        node = graph.get_node(node_id)
        assert node is not None and node.run_id == "owner"
        assert node.worktree_path == str(worktree) and worktree.exists()
    finally:
        graph.close()


def test_reviewer_run_stopped_does_not_confirm_worker(
    tmp_path: Path, worker: subprocess.Popen[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, node_id, worktree = _owned_graph(tmp_path, worker)

    class ReviewManager:
        def __init__(self) -> None:
            self.emitter: QueueEmitter | None = None

        def create_run(self, config: RunConfig, emitter: QueueEmitter) -> object:
            _ = config
            self.emitter = emitter
            return SimpleNamespace(state=SimpleNamespace(run_id="reviewer"))

        def start_run(self, run_id: str) -> None:
            assert self.emitter is not None
            self.emitter.queue.put(Event(EventType.RUN_STOPPED, run_id, NoData()))

    monkeypatch.setattr("milknado.adapters.loop.RunManager", ReviewManager)
    try:
        adapter = LoopAdapter(graph=graph)
        with pytest.raises(PreservedWorkerRun) as preserved:
            _ = adapter.run_node_review(
                "agent",
                "review",
                worktree,
                worktree,
                timeout_seconds=0.1,
                graph_run_id="owner",
            )
        assert preserved.value.node_id == node_id
        assert preserved.value.run_id == "owner"
        node = graph.get_node(node_id)
        assert node is not None and node.run_id == "owner"
        assert node.worktree_path == str(worktree) and worktree.exists()
    finally:
        graph.close()


def test_executor_does_not_discard_unconfirmed_worker_worktree(
    tmp_path: Path, worker: subprocess.Popen[str]
) -> None:
    graph, node_id, worktree = _owned_graph(tmp_path, worker)
    git = MagicMock()
    executor = Executor(
        graph,
        cast(GitPort, git),
        cast(LoopPort, MagicMock()),
        cast(CrgPort, MagicMock()),
    )
    pool = FlockSlotPool(1)
    executor.use_host_capacity(pool)
    executor._slots.take("owner", node_id, tmp_path, wait=False)  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(PreservedWorkerRun):
            executor.fail(node_id)
        with pytest.raises(PreservedWorkerRun):
            executor.finish_preserved_abort(node_id, "owner", "reviewer")
        git.remove_worktree.assert_not_called()  # pyright: ignore[reportAny]
        with pytest.raises(HostCapacityFull):
            _ = pool.acquire("other", node_id + 1, tmp_path)
        node = graph.get_node(node_id)
        assert node is not None and node.status is NodeStatus.RUNNING
        assert node.run_id == "owner" and node.worktree_path == str(worktree)
        assert worktree.exists()
    finally:
        executor._slots.drop(node_id)  # pyright: ignore[reportPrivateUsage]
        graph.close()


def test_unconfirmed_record_blocks_blocked_node_admission(
    tmp_path: Path, worker: subprocess.Popen[str]
) -> None:
    graph, node_id, worktree = _owned_graph(tmp_path, worker, blocked=True)
    try:
        assert graph.claim_node(node_id, "new-owner", now="2026-01-01T00:01:00+00:00") is False
        with pytest.raises(InvalidTransition):
            graph.mark_running(node_id, str(worktree), "new-branch", "new-owner")
        node = graph.get_node(node_id)
        assert node is not None and node.status is NodeStatus.BLOCKED
        assert node.run_id == "owner" and node.worktree_path == str(worktree)
    finally:
        graph.close()
