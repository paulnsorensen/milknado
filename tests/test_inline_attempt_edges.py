"""Configured inline retries at timeout, startup, cancellation, and poll boundaries."""

from __future__ import annotations

import os
import shutil
import signal
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import psutil
import pytest

import milknado.loop._process_identity as process_identity
from milknado.domains.common import IdentityState, NodeStatus, WorktreeMode, identity_state
from milknado.mcp._core import NodeSummary, RunDict
from milknado.mcp.run import (
    milknado_run_cancel,
    milknado_run_inline,
    milknado_run_inline_poll,
    milknado_run_inline_start,
)
from milknado.mcp.server import open_graph
from milknado.mcp.todo_mutate import milknado_todo_add
from tests.test_run_inline_worktree import (
    _call,  # pyright: ignore[reportPrivateUsage]
    _init_repo,  # pyright: ignore[reportPrivateUsage]
    _wait_for_terminal,  # pyright: ignore[reportPrivateUsage]
)


def _bounded_task(root: Path) -> int:
    _init_repo(root)
    _ = (root / "milknado.toml").write_text("[milknado]\nmax_node_attempts = 2\n")
    task = cast(
        NodeSummary,
        _call(milknado_todo_add, description="edge", kind="task", project_root=str(root)),
    )
    return int(task["id"])


def _attempt_state(root: Path, node_id: int) -> tuple[NodeStatus, int]:
    graph, _ = open_graph(root)
    try:
        node = graph.get_node(node_id)
        assert node is not None
        return node.status, graph.task_attempts.failed_count(node_id)
    finally:
        graph.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_inline_timeout_counts_confirmed_execution_once(
    tmp_path: Path, worker_stub: Callable[[str], str], asynchronous: bool
) -> None:
    root = tmp_path / "repo"
    node_id = _bounded_task(root)
    arguments = {
        "node_id": node_id,
        "worker_cmd": worker_stub("sh -c 'sleep 3'"),
        "timeout_seconds": 1,
        "worktree": WorktreeMode.THIS_BRANCH,
        "project_root": str(root),
    }
    if asynchronous:
        started = cast(RunDict, _call(milknado_run_inline_start, **arguments))
        result = _wait_for_terminal(cast(str, started["run_id"]), str(root))
        assert (
            _call(milknado_run_inline_poll, run_id=started["run_id"], project_root=str(root))
            == result
        )
    else:
        result = cast(RunDict, _call(milknado_run_inline, **arguments))
    assert result["status"] == ("failed" if asynchronous else "pending")
    assert result["timed_out"] is True
    assert _attempt_state(root, node_id) == (NodeStatus.PENDING, 1)


def _child_identity(path: Path) -> tuple[int, float]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            pid = int(path.read_text(encoding="utf-8"))
            return pid, psutil.Process(pid).create_time()
        except (FileNotFoundError, ValueError, psutil.NoSuchProcess):
            time.sleep(0.02)
    raise AssertionError("worker descendant did not start")


def _assert_worker_closed_and_requeued(root: Path, run_id: str, node_id: int) -> None:
    graph, _ = open_graph(root)
    try:
        record = graph.runs.get_worker(run_id)
        assert record is not None and record.ended_at is not None
        assert graph.runs.live_workers(run_id=run_id) == ()
        assert graph.task_attempts.failed_count(node_id) == 1
        node = graph.get_node(node_id)
        assert node is not None and node.status is NodeStatus.PENDING
    finally:
        graph.close()


def _assert_owner_kept(root: Path, node_id: int) -> tuple[int, float] | None:
    graph, _ = open_graph(root)
    try:
        node = graph.get_node(node_id)
        assert node is not None and node.status is NodeStatus.RUNNING
        assert graph.task_attempts.failed_count(node_id) == 0
        runs = graph.runs.for_node(node_id)
        assert len(runs) == 1 and runs[0]["status"] == "running"
        workers = graph.runs.live_workers(run_id=runs[0]["run_id"])
        assert len(workers) == 1
        if workers[0].helper_pid is not None and workers[0].helper_start_token is not None:
            return workers[0].helper_pid, workers[0].helper_start_token
        return None
    finally:
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX worker-tree cleanup")
@pytest.mark.parametrize("asynchronous", [False, True])
def test_inline_timeout_closes_descendant_before_retry(
    tmp_path: Path, worker_stub: Callable[[str], str], asynchronous: bool
) -> None:
    root = tmp_path / "repo"
    node_id = _bounded_task(root)
    arguments = {
        "node_id": node_id,
        "worker_cmd": worker_stub("sh -c 'sleep 30 & echo $! > child.pid; wait'"),
        "timeout_seconds": 2,
        "worktree": WorktreeMode.THIS_BRANCH,
        "project_root": str(root),
    }
    pid = 0
    token = 0.0
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            if asynchronous:
                started = cast(RunDict, _call(milknado_run_inline_start, **arguments))
                run_id = cast(str, started["run_id"])
                pid, token = _child_identity(root / "child.pid")
                result = _wait_for_terminal(run_id, str(root))
            else:
                future = pool.submit(_call, milknado_run_inline, **arguments)
                pid, token = _child_identity(root / "child.pid")
                result = cast(RunDict, future.result(timeout=12))
                run_id = cast(str, result["run_id"])
        assert identity_state(pid, token) == "gone"
        _assert_worker_closed_and_requeued(root, run_id, node_id)
        assert result["timed_out"] is True
    finally:
        if pid and identity_state(pid, token) == "live":
            os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(os.name == "nt", reason="POSIX worker-tree cleanup")
def test_inline_unresolved_cleanup_keeps_owner(
    tmp_path: Path, worker_stub: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    node_id = _bounded_task(root)

    def unresolved_group(_pgid: int) -> IdentityState:
        return "unknown"

    monkeypatch.setattr(process_identity, "group_state", unresolved_group)
    pid = 0
    token = 0.0
    helper: tuple[int, float] | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _call,
                milknado_run_inline,
                node_id=node_id,
                worker_cmd=worker_stub("sh -c 'sleep 30 & echo $! > child.pid; wait'"),
                timeout_seconds=2,
                worktree=WorktreeMode.THIS_BRANCH,
                project_root=str(root),
            )
            pid, token = _child_identity(root / "child.pid")
            with pytest.raises(
                RuntimeError, match="worker cleanup unresolved|lifeline helper exit unconfirmed"
            ):
                _ = future.result(timeout=15)
        helper = _assert_owner_kept(root, node_id)
    finally:
        if pid and identity_state(pid, token) == "live":
            os.kill(pid, signal.SIGKILL)
        if helper is not None and identity_state(*helper) == "live":
            os.kill(helper[0], signal.SIGKILL)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_inline_missing_target_is_startup_failure_without_count(
    tmp_path: Path, asynchronous: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    node_id = _bounded_task(root)
    shim = shutil.which("crush")
    assert shim is not None
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            entry
            for entry in os.environ["PATH"].split(os.pathsep)
            if entry != str(Path(shim).parent)
        ),
    )
    assert shutil.which("crush") is None
    arguments = {
        "node_id": node_id,
        "worker_cmd": "crush",
        "worktree": WorktreeMode.THIS_BRANCH,
        "project_root": str(root),
    }
    if asynchronous:
        started = cast(RunDict, _call(milknado_run_inline_start, **arguments))
        final = _wait_for_terminal(cast(str, started["run_id"]), str(root))
        assert final["status"] == "failed"
    else:
        with pytest.raises(OSError):
            _ = _call(milknado_run_inline, **arguments)
    assert _attempt_state(root, node_id) == (NodeStatus.FAILED, 0)


def test_async_inline_cancel_and_poll_do_not_requeue(
    tmp_path: Path, worker_stub: Callable[[str], str]
) -> None:
    root = tmp_path / "repo"
    node_id = _bounded_task(root)
    started = cast(
        RunDict,
        _call(
            milknado_run_inline_start,
            node_id=node_id,
            worker_cmd=worker_stub("sh -c 'sleep 5'"),
            worktree=WorktreeMode.THIS_BRANCH,
            project_root=str(root),
        ),
    )
    run_id = started["run_id"]
    assert run_id is not None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        graph, _ = open_graph(root)
        try:
            attempt = graph.task_attempts.get(run_id)
            record = graph.runs.get(run_id)
            # Release records "executed" before the run row receives the worker pid.
            # Cancel needs the pid, so wait for both.
            if (
                attempt is not None
                and attempt.start_state == "executed"
                and record is not None
                and record["pid"] is not None
            ):
                break
        finally:
            graph.close()
        time.sleep(0.02)
    else:
        raise AssertionError("worker never confirmed command execution")
    _ = _call(milknado_run_cancel, run_id=run_id, project_root=str(root))
    final = _wait_for_terminal(run_id, str(root))
    assert final["status"] == "failed"
    assert _call(milknado_run_inline_poll, run_id=run_id, project_root=str(root)) == final
    assert _attempt_state(root, node_id) == (NodeStatus.FAILED, 0)
