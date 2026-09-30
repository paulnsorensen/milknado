from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions
from milknado.loop._process_lifecycle import spawn_protected


@pytest.mark.skipif(os.name == "nt", reason="POSIX helper uses passed file descriptors")
def test_shutdown_cancels_held_replacement_ready_without_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id)
    context = ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path)
    options = SpawnOptions(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        tmp_path,
        None,
        False,
        subprocess.DEVNULL,
        subprocess.PIPE,
        subprocess.PIPE,
    )
    protected = spawn_protected(options, context)
    fake = tmp_path / "held-helper"
    _ = fake.write_text("#!/bin/sh\nexec sleep 30\n")
    fake.chmod(0o700)
    replacement_pid = 0
    try:
        initial = graph.runs.get_worker(protected.identity.invocation_id)
        assert initial is not None and initial.helper_pid is not None
        monkeypatch.setattr(sys, "executable", str(fake))
        os.kill(initial.helper_pid, signal.SIGKILL)
        limit = time.monotonic() + 4
        while time.monotonic() < limit:
            record = graph.runs.get_worker(protected.identity.invocation_id)
            if record is not None and record.helper_generation == 1 and record.helper_pid:
                replacement_pid = record.helper_pid
                break
            time.sleep(0.02)
        assert replacement_pid
        deadline = time.monotonic() + 2.5
        _ = protected.shutdown(deadline)
        assert time.monotonic() < deadline + 0.3
        assert protected.process.poll() is not None and (
            not psutil.pid_exists(replacement_pid)
            or psutil.Process(replacement_pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        protected.close_lifeline()
        if protected.process.poll() is None:
            os.killpg(protected.process.pid, signal.SIGKILL)
        _ = protected.process.wait(timeout=2)
        if replacement_pid and psutil.pid_exists(replacement_pid):
            helper = psutil.Process(replacement_pid)
            if helper.status() != psutil.STATUS_ZOMBIE:
                helper.kill()
        graph.close()
