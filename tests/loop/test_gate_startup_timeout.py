"""A gate that never reports ready is a startup failure, not an unknown launch."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from milknado.loop import _process_gate
from milknado.loop._process_gate import SpawnOptions, StartEvidence, spawn_gated

pytestmark = [
    pytest.mark.real_process,
    pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate is required"),
]

_STALL_STARTUP = "import time\ntime.sleep(300)\n"


def test_stalled_gate_startup_is_confirmed_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inject = tmp_path / "inject"
    inject.mkdir()
    _ = (inject / "sitecustomize.py").write_text(_STALL_STARTUP, encoding="utf-8")
    sentinel = tmp_path / "worker-ran"
    monkeypatch.setattr(_process_gate, "_GATE_STARTUP_TIMEOUT", 0.5)
    env = {**os.environ, "PYTHONPATH": str(inject)}
    command = (sys.executable, "-c", f"open({str(sentinel)!r}, 'w').close()")
    worker = spawn_gated(
        SpawnOptions(command, tmp_path, env, True, subprocess.DEVNULL, subprocess.PIPE, None)
    )
    try:
        started = time.monotonic()
        worker.release()
        elapsed = time.monotonic() - started
        assert elapsed < 10
        assert worker.start_evidence is StartEvidence.STARTUP_FAILED
        assert worker.process.poll() is not None
        assert not sentinel.exists()
    finally:
        worker.process.kill()
        _ = worker.process.wait(timeout=5)
        worker.close_gate()
        for pipe in (worker.process.stdout, worker.process.stdin):
            if pipe is not None:
                pipe.close()
