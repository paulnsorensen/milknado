from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from milknado.domains.graph import MikadoGraph
from milknado.loop._agent import _ResolvedAgentRun, _run_agent_blocking, _run_agent_streaming
from milknado.loop._process_lifecycle import ProtectionContext, SpawnOptions, spawn_protected


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_protected_invocation_preserves_output_and_closes_record(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    options = SpawnOptions(
        (sys.executable, "-c", "print('ready-output', flush=True)"),
        tmp_path, None, True, subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE,
    )
    try:
        worker = spawn_protected(options, ProtectionContext(graph.runs, "run-1", db_path))
        assert worker.process.stdout is not None
        assert worker.process.stdout.read().strip() == "ready-output"
        assert worker.process.wait(timeout=5) == 0
        assert worker.finish(timeout=5)
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_blocking_agent_uses_protected_worker_without_changing_output(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    context = ProtectionContext(graph.runs, "run-1", graph.db_path)
    try:
        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-c", "print('agent-output', flush=True)"],
                None, timeout=5, log_dir=tmp_path, iteration=1,
                spawn_worker=lambda options: spawn_protected(options, context),
            )
        )
        assert result.returncode == 0
        assert result.captured_stdout == "agent-output\n"
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_streaming_agent_preserves_json_framing(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    context = ProtectionContext(graph.runs, "run-1", graph.db_path)
    try:
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                [sys.executable, "-c", "print('{\"type\":\"result\",\"result\":\"streamed\"}', flush=True)"],
                None, timeout=5, log_dir=tmp_path, iteration=1,
                spawn_worker=lambda options: spawn_protected(options, context),
            )
        )
        assert result.returncode == 0
        assert result.result_text == "streamed"
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_dead_lifeline_is_replaced_without_restarting_worker(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = spawn_protected(
        SpawnOptions(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            tmp_path, None, False, subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE,
        ),
        ProtectionContext(graph.runs, "run-1", graph.db_path),
    )
    try:
        before = graph.runs.get_worker(worker.identity.invocation_id)
        assert before is not None and before.helper_pid is not None
        os.kill(before.helper_pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            after = graph.runs.get_worker(worker.identity.invocation_id)
            if after is not None and after.ready_generation == 1:
                break
            time.sleep(0.05)
        else:
            pytest.fail("replacement lifeline did not become ready")
        assert after is not None and after.helper_pid != before.helper_pid
        assert after.pid == before.pid == worker.process.pid
        assert worker.process.poll() is None
        os.kill(worker.process.pid, signal.SIGTERM)
        _ = worker.process.wait(timeout=5)
        assert worker.finish(timeout=5)
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        if worker.process.poll() is None:
            os.killpg(worker.process.pid, signal.SIGKILL)
            _ = worker.process.wait(timeout=1)
        graph.close()
