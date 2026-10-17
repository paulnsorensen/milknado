"""The exec gate reports a worker command that cannot be executed."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.real_process


def test_missing_worker_command_reports_failure_on_the_status_pipe() -> None:
    gate_read, gate_write = os.pipe()
    status_read, status_write = os.pipe()
    _ = os.write(gate_write, b"R")
    os.close(gate_write)
    argv = [
        sys.executable,
        "-m",
        "milknado.loop._exec_gate",
        str(gate_read),
        str(status_write),
        "/no/such/cmd",
    ]
    process = subprocess.Popen(argv, pass_fds=(gate_read, status_write))
    os.close(gate_read)
    os.close(status_write)

    exit_code = process.wait(timeout=30)
    with os.fdopen(status_read, "rb") as status:
        reported = status.read()

    assert exit_code == 72
    assert reported == b"GPF"
