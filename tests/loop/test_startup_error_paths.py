"""Failure paths for gated worker acquisition."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions, spawn_gated
from milknado.loop._process_lifecycle import spawn_protected
from milknado.loop._process_registry import WorkerRegistry


def _options(tmp_path: Path, command: tuple[str, ...]) -> SpawnOptions:
    return SpawnOptions(
        command, tmp_path, None, True, subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE
    )


def test_empty_command_fails_before_process_creation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="worker command is empty"):
        _ = spawn_gated(_options(tmp_path, ()))


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")
def test_popen_failure_closes_both_gate_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_pipe = os.pipe
    gate_fds: list[int] = []

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        gate_fds.extend(pair)
        return pair

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise OSError("spawn denied")

    monkeypatch.setattr(os, "pipe", tracked_pipe)
    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    with pytest.raises(OSError, match="spawn denied"):
        _ = spawn_gated(_options(tmp_path, (sys.executable, "-c", "pass")))
    assert len(gate_fds) == 2
    for fd in gate_fds:
        with pytest.raises(OSError):
            _ = os.fstat(fd)


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")
def test_identity_failure_reaps_child_and_closes_parent_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_popen = subprocess.Popen
    children: list[subprocess.Popen[str]] = []
    marker = tmp_path / "ran"

    def captured_popen(*args: object, **kwargs: object) -> subprocess.Popen[str]:
        child = real_popen(*args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]
        children.append(child)
        return child

    def fail_identity(_pid: int) -> None:
        raise RuntimeError("identity unavailable")

    monkeypatch.setattr(subprocess, "Popen", captured_popen)
    monkeypatch.setattr(psutil, "Process", fail_identity)
    try:
        command = (
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        )
        with pytest.raises(RuntimeError, match="identity unavailable"):
            _ = spawn_gated(_options(tmp_path, command))
        assert len(children) == 1
        child = children[0]
        assert child.poll() is not None
        assert child.stdout is not None and child.stdout.closed
        assert child.stderr is not None and child.stderr.closed
        assert not marker.exists()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                _ = child.wait(timeout=1)
            if child.stdout is not None:
                child.stdout.close()
            if child.stderr is not None:
                child.stderr.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required")
def test_registry_ticket_closes_when_spawn_fails(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    registry = WorkerRegistry()
    current = psutil.Process()
    owner = WorkerOwner("run", current.pid, current.create_time(), "run", node.id)
    context = ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path, registry)
    try:
        with pytest.raises(ValueError, match="worker command is empty"):
            _ = spawn_protected(_options(tmp_path, ()), context)
        assert registry.tickets == set()
    finally:
        graph.close()


def test_exec_gate_rejects_missing_arguments() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "milknado.loop._exec_gate"],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 72
