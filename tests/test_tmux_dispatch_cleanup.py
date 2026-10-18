"""Real-tmux cleanup, cancel, exactly-once evidence, and unresolved-worker witnesses."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

from milknado.adapters.tmux import TmuxAdapter
from milknado.domains.common import NodeStatus, pid_alive
from milknado.domains.dispatch import runs_dir
from milknado.domains.dispatch.tmux_run import read_exec_evidence
from milknado.mcp.run import milknado_run_inline_start
from milknado.mcp.server import open_graph
from tests.test_tmux_dispatch import (
    _add_task,  # pyright: ignore[reportPrivateUsage]
    _call,  # pyright: ignore[reportPrivateUsage]
    _node_status,  # pyright: ignore[reportPrivateUsage]
    _wait_for_terminal,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.real_process


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux binary not available")
def test_real_inline_tmux_failure_uses_target_receipt_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_stub: Callable[[str], str]
) -> None:
    socket = Path("/tmp") / f"milknado-inline-{tmp_path.name}.sock"
    adapter = TmuxAdapter(tmp_path, socket_path=socket, config_path=Path("/dev/null"))

    def adapter_for_root(_root: Path) -> TmuxAdapter:
        return adapter

    monkeypatch.setattr("milknado.app.run.TmuxAdapter", adapter_for_root)
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", adapter_for_root)
    _ = (tmp_path / "milknado.toml").write_text("[milknado]\nmax_node_attempts = 2\n")
    root = str(tmp_path)
    node_id = _add_task(root)
    try:
        for number, status in ((1, NodeStatus.PENDING), (2, NodeStatus.BLOCKED)):
            started = _call(
                milknado_run_inline_start,
                node_id=node_id,
                worker_cmd=worker_stub("sh -c 'exit 7'"),
                use_tmux=True,
                project_root=root,
            )
            final = _wait_for_terminal(root, started["run_id"])
            assert final["status"] == "failed"
            assert _node_status(root, node_id) is status
            graph, _ = open_graph(tmp_path)
            try:
                assert graph.task_attempts.failed_count(node_id) == number
                nonce = graph.runs.latest_message(started["run_id"], "tmux_exec_nonce")
                assert nonce is not None
                receipt = runs_dir(tmp_path) / f"{started['run_id']}.{nonce}.exec.json"
                assert read_exec_evidence(receipt, nonce) == "executed"
            finally:
                graph.close()
    finally:
        _ = subprocess.run(
            ["tmux", "-S", str(socket), "kill-server"], capture_output=True, check=False
        )


def _real_adapter(adapter: TmuxAdapter) -> Callable[[Path], TmuxAdapter]:
    def for_root(_root: Path) -> TmuxAdapter:
        return adapter

    return for_root


def _await_covered_descendant(root: Path, run_id: str, child_file: Path) -> int:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if child_file.exists():
            child_pid = int(child_file.read_text())
            graph, _ = open_graph(root)
            try:
                workers = graph.runs.live_workers(run_id=run_id)
                if any(pid == child_pid for row in workers for pid, _, _ in row.descendants):
                    return child_pid
            finally:
                graph.close()
        time.sleep(0.05)
    raise AssertionError("detached child was not durably observed")


def _assert_requeued_without_workers(root: Path, run_id: str, node_id: int) -> None:
    graph, _ = open_graph(root)
    try:
        assert graph.runs.live_workers(run_id=run_id) == ()
        assert graph.task_attempts.failed_count(node_id) == 1
    finally:
        graph.close()


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux binary not available")
def test_real_tmux_failure_cleans_detached_descendant_before_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_stub: Callable[[str], str]
) -> None:
    socket = Path("/tmp") / f"milknado-tree-{tmp_path.name}.sock"
    adapter = TmuxAdapter(tmp_path, socket_path=socket, config_path=Path("/dev/null"))
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", _real_adapter(adapter))
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", _real_adapter(adapter))
    _ = (tmp_path / "milknado.toml").write_text("[milknado]\nmax_node_attempts = 2\n")
    node_id = _add_task(str(tmp_path))
    child_file = tmp_path / "child.pid"
    release = tmp_path / "release"
    command = worker_stub(
        "sh -c 'setsid sleep 120 </dev/null >/dev/null 2>&1 & "
        + f"echo $! > {child_file}; while [ ! -e {release} ]; do sleep 0.05; done; exit 7'"
    )
    child_pid: int | None = None
    try:
        started = _call(
            milknado_run_inline_start,
            node_id=node_id,
            worker_cmd=command,
            use_tmux=True,
            project_root=str(tmp_path),
        )
        child_pid = _await_covered_descendant(tmp_path, started["run_id"], child_file)
        assert _node_status(str(tmp_path), node_id) is NodeStatus.RUNNING
        _ = release.touch()
        final = _wait_for_terminal(str(tmp_path), started["run_id"])
        assert final["status"] == "failed"
        assert final["exit_code"] == 7
        assert _node_status(str(tmp_path), node_id) is NodeStatus.PENDING
        assert not pid_alive(child_pid)
        _assert_requeued_without_workers(tmp_path, started["run_id"], node_id)
    finally:
        if child_pid is None and child_file.exists():
            child_pid = int(child_file.read_text())
        if child_pid is not None and psutil.pid_exists(child_pid):
            with suppress(ProcessLookupError):
                os.killpg(child_pid, signal.SIGKILL)
        _ = subprocess.run(["tmux", "-S", str(socket), "kill-server"], check=False)


def _assert_cancelled_without_workers(root: Path, run_id: str, node_id: int) -> None:
    graph, _ = open_graph(root)
    try:
        assert graph.runs.live_workers(run_id=run_id) == ()
        assert graph.task_attempts.failed_count(node_id) == 0
    finally:
        graph.close()


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux binary not available")
def test_real_tmux_cancel_cleans_covered_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_stub: Callable[[str], str]
) -> None:
    from milknado.mcp.run import milknado_run_cancel

    socket = Path("/tmp") / f"milknado-cancel-{tmp_path.name}.sock"
    adapter = TmuxAdapter(tmp_path, socket_path=socket, config_path=Path("/dev/null"))
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", _real_adapter(adapter))
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", _real_adapter(adapter))
    node_id = _add_task(str(tmp_path))
    child_file = tmp_path / "cancel-child.pid"
    command = worker_stub(
        "sh -c 'setsid sleep 120 </dev/null >/dev/null 2>&1 & "
        + f"echo $! > {child_file}; while :; do sleep 0.05; done'"
    )
    child_pid: int | None = None
    try:
        started = _call(
            milknado_run_inline_start,
            node_id=node_id,
            worker_cmd=command,
            use_tmux=True,
            project_root=str(tmp_path),
        )
        child_pid = _await_covered_descendant(tmp_path, started["run_id"], child_file)
        final = _call(milknado_run_cancel, run_id=started["run_id"], project_root=str(tmp_path))
        assert final["status"] == "failed"
        assert _node_status(str(tmp_path), node_id) is NodeStatus.FAILED
        assert not pid_alive(child_pid)
        _assert_cancelled_without_workers(tmp_path, started["run_id"], node_id)
    finally:
        if child_pid is None and child_file.exists():
            child_pid = int(child_file.read_text())
        if child_pid is not None and psutil.pid_exists(child_pid):
            with suppress(ProcessLookupError):
                os.killpg(child_pid, signal.SIGKILL)
        _ = subprocess.run(["tmux", "-S", str(socket), "kill-server"], check=False)


def _wait_for(condition: Callable[[], bool], diagnostics: Callable[[], str] | None = None) -> None:
    deadline = time.monotonic() + 60
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert condition(), diagnostics() if diagnostics else "condition not met within 60s"


def _record_unresolved_peer(root: Path, run_node: tuple[str, int], peer_pid: int) -> None:
    from milknado.domains.common import WorkerIdentity, WorkerOwner
    from milknado.domains.graph import WorkerEvidenceStore

    run_id, node_id = run_node
    peer = psutil.Process(peer_pid)
    graph, config = open_graph(root)
    graph.close()
    owner = WorkerOwner(run_id, os.getpid(), psutil.Process().create_time(), run_id, node_id)
    identity = WorkerIdentity("unresolved-peer", peer.pid, peer.pid, peer.create_time())
    with WorkerEvidenceStore(config.db_path) as store:
        store.record_worker(owner, identity)


def _assert_unresolved_worker_kept(root: Path, run_id: str, node_id: int) -> None:
    graph, _ = open_graph(root)
    try:
        assert graph.task_attempts.failed_count(node_id) == 0
        assert not graph.release(node_id, run_id)
        assert [row.invocation_id for row in graph.runs.live_workers(run_id=run_id)] == [
            "unresolved-peer"
        ]
    finally:
        graph.close()


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux binary not available")
def test_real_tmux_known_unresolved_worker_blocks_release_and_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_stub: Callable[[str], str]
) -> None:
    socket = Path("/tmp") / f"milknado-unresolved-{tmp_path.name}.sock"
    adapter = TmuxAdapter(tmp_path, socket_path=socket, config_path=Path("/dev/null"))
    monkeypatch.setattr("milknado.app.run.TmuxAdapter", _real_adapter(adapter))
    monkeypatch.setattr("milknado.mcp.run.TmuxAdapter", _real_adapter(adapter))
    _ = (tmp_path / "milknado.toml").write_text("[milknado]\nmax_node_attempts = 2\n")
    node_id = _add_task(str(tmp_path))
    started_file = tmp_path / "target.started"
    release = tmp_path / "release"
    command = worker_stub(
        f"sh -c 'touch {started_file}; while [ ! -e {release} ]; do sleep 0.05; done; exit 7'"
    )
    unresolved = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        started = _call(
            milknado_run_inline_start,
            node_id=node_id,
            worker_cmd=command,
            use_tmux=True,
            project_root=str(tmp_path),
        )
        run_id = started["run_id"]
        _wait_for(started_file.exists)
        _record_unresolved_peer(tmp_path, (run_id, node_id), unresolved.pid)
        _ = release.touch()
        log = Path(started["log_path"])
        _wait_for(
            lambda: "worker cleanup unresolved" in log.read_text(),
            lambda: f"log tail: {log.read_text()[-2000:]!r}",
        )
        time.sleep(0.5)
        assert _node_status(str(tmp_path), node_id) is NodeStatus.RUNNING
        _assert_unresolved_worker_kept(tmp_path, run_id, node_id)
    finally:
        unresolved.kill()
        _ = unresolved.wait()
        _ = subprocess.run(["tmux", "-S", str(socket), "kill-server"], check=False)
