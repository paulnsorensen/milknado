from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

import milknado.loop._agent as agent
import milknado.loop._process_lifecycle as lifecycle
import milknado.loop._process_observation as observation
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._agent import _ResolvedAgentRun, _run_agent_blocking, _run_agent_streaming
from milknado.loop._process_lifecycle import ProtectionContext, SpawnOptions, spawn_protected


def _context(graph: MikadoGraph, node_id: int) -> ProtectionContext:
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node_id)
    return ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path)

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
        worker = spawn_protected(options, _context(graph, node.id))
        assert worker.process.stdout is not None
        assert worker.process.stdout.read().strip() == "ready-output"
        assert worker.process.wait(timeout=5) == 0
        assert worker.finish(timeout=5)
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_shared_owner_stops_worker_and_drains_its_pipes(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = spawn_protected(
        SpawnOptions(
            (sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"),
            tmp_path, None, True, subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE,
        ),
        _context(graph, node.id),
    )
    assert worker.process.stdout is not None
    output: list[str] = []
    reader = threading.Thread(target=lambda: output.append(worker.process.stdout.read()))
    reader.start()
    try:
        assert worker.cleanup((reader,), deadline=time.monotonic() + 4)
        assert not reader.is_alive()
        assert worker.process.stdout.closed
        assert worker.process.stderr is not None and worker.process.stderr.closed
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        if worker.process.poll() is None:
            worker.process.kill()
            _ = worker.process.wait(timeout=1)
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_blocking_agent_uses_protected_worker_without_changing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def legacy_cleanup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("protected worker used legacy cleanup")

    monkeypatch.setattr(agent, "_cleanup_agent", legacy_cleanup)
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    context = _context(graph, node.id)
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
def test_streaming_agent_preserves_json_framing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def legacy_cleanup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("protected worker used legacy cleanup")

    monkeypatch.setattr(agent, "_cleanup_agent", legacy_cleanup)
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    context = _context(graph, node.id)
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
        _context(graph, node.id),
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

@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_failed_observation_stops_worker_and_keeps_open_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    worker = spawn_protected(
        SpawnOptions(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            tmp_path, None, False, subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE,
        ),
        _context(graph, node.id),
    )
    try:
        def fail_observation(_identity):
            raise RuntimeError("simulated observation failure")

        monkeypatch.setattr(observation, "observe_descendants", fail_observation)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and worker.process.poll() is None:
            time.sleep(0.05)
        assert worker.process.poll() is not None
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.observation_owner == "supervisor"
        assert record.ended_at is None
    finally:
        if worker.process.poll() is None:
            os.killpg(worker.process.pid, signal.SIGKILL)
            _ = worker.process.wait(timeout=1)
        graph.close()
