"""Real target-exec receipts, not pane or wrapper launch guesses."""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from milknado.domains.common import default_config
from milknado.domains.dispatch.tmux_run import read_exec_evidence
from milknado.domains.graph import MikadoGraph, StartTaskAttempt


def _wrapper(path: Path, *command: str) -> list[str]:
    root = path.parent
    run_id = f"receipt-{secrets.token_hex(8)}"
    db_path = default_config(root).db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    graph = MikadoGraph(db_path)
    try:
        node = graph.add_node("receipt target")
        assert graph.claim_node(node.id, run_id, now="2026-01-01T00:00:00+00:00")
        _ = graph.task_attempts.start(StartTaskAttempt(node.id, run_id, run_id, run_id))
        graph.runs.start(
            run_id, node.id, str(path), "2026-01-01T00:00:00+00:00", 10, attempt_id=run_id
        )
    finally:
        graph.close()
    return [
        "env",
        f"MILKNADO_PROJECT_ROOT={root}",
        f"MILKNADO_RUN_ID={run_id}",
        f"MILKNADO_NODE_ID={node.id}",
        sys.executable,
        "-m",
        "milknado.adapters._tmux_exec",
        "current-nonce",
        str(path),
        "-",
        *command,
    ]


def test_tmux_receipt_confirms_normal_target_exit(tmp_path: Path) -> None:
    receipt = tmp_path / "exec.json"
    result = subprocess.run(
        _wrapper(receipt, sys.executable, "-c", "raise SystemExit(7)"),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 7
    assert read_exec_evidence(receipt, "current-nonce") == "executed"
    assert read_exec_evidence(receipt, "old-nonce") is None
    assert json.loads(receipt.read_text())["returncode"] == 7


def test_tmux_receipt_rejects_preexec_failure_and_signal(tmp_path: Path) -> None:
    receipt = tmp_path / "exec.json"
    result = subprocess.run(
        _wrapper(receipt, "/definitely/not/a/worker"),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 125
    assert read_exec_evidence(receipt, "current-nonce") == "startup_failed"
    result = subprocess.run(
        _wrapper(
            receipt,
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 128 + signal.SIGTERM
    assert read_exec_evidence(receipt, "current-nonce") is None


def test_tmux_receipt_live_identity_and_malformed_data(tmp_path: Path) -> None:
    receipt = tmp_path / "exec.json"
    process = subprocess.Popen(
        _wrapper(receipt, sys.executable, "-c", "import time; time.sleep(0.7)"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if read_exec_evidence(receipt, "current-nonce") == "executed":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("live target receipt was not observed")
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            _ = process.wait()
    _ = receipt.write_text('{"nonce":"current-nonce","state":"EXECUTED","pid":1}')
    assert read_exec_evidence(receipt, "current-nonce") is None
    _ = receipt.write_bytes(os.urandom(5000))
    assert read_exec_evidence(receipt, "current-nonce") is None


def test_tmux_receipt_rejects_reused_pid_token(tmp_path: Path) -> None:
    receipt = tmp_path / "exec.json"
    pid = os.getpid()
    token = psutil.Process(pid).create_time()
    _ = receipt.write_text(
        json.dumps(
            {"nonce": "current-nonce", "state": "EXECUTED", "pid": pid, "start_token": token}
        )
    )
    assert read_exec_evidence(receipt, "current-nonce") == "executed"
    _ = receipt.write_text(
        json.dumps(
            {"nonce": "current-nonce", "state": "EXECUTED", "pid": pid, "start_token": token + 1}
        )
    )
    assert read_exec_evidence(receipt, "current-nonce") is None


@pytest.mark.skipif(os.name == "nt", reason="Windows does not retain zombie child processes")
def test_tmux_receipt_rejects_dead_target_with_matching_token(tmp_path: Path) -> None:
    receipt = tmp_path / "exec.json"
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        target = psutil.Process(process.pid)
        token = target.create_time()
        deadline = time.monotonic() + 2
        while target.status() not in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}:
            assert time.monotonic() < deadline, "target did not become inactive"
            time.sleep(0.01)
        _ = receipt.write_text(
            json.dumps(
                {
                    "nonce": "current-nonce",
                    "state": "EXECUTED",
                    "pid": process.pid,
                    "start_token": token,
                }
            )
        )
        assert read_exec_evidence(receipt, "current-nonce") is None
    finally:
        assert process.wait(timeout=2) == 0


def test_tmux_receipt_rejects_ambiguous_and_invalid_fields(tmp_path: Path) -> None:
    receipt = tmp_path / "exec.json"
    token = psutil.Process().create_time()
    records = (
        {"state": "EXECUTED", "pid": True, "start_token": token},
        {"state": "EXECUTED", "pid": 0, "start_token": token},
        {"state": "EXECUTED", "pid": os.getpid(), "start_token": None},
        {"state": "EXECUTED", "pid": os.getpid(), "start_token": token, "returncode": 0},
        {"state": "FINISHED", "pid": os.getpid(), "returncode": True},
        {"state": "FINISHED", "pid": os.getpid(), "returncode": -signal.SIGTERM},
        {"state": "EXEC_FAILED", "error": "failure", "pid": os.getpid()},
        {"state": "PREPARED", "pid": os.getpid()},
    )
    for fields in records:
        _ = receipt.write_text(json.dumps({"nonce": "current-nonce", **fields}))
        assert read_exec_evidence(receipt, "current-nonce") is None, fields
