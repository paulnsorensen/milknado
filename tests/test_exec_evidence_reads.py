"""Exec receipt reads stay bounded and never follow special files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Thread

import pytest

from milknado.domains.dispatch.tmux_run import read_exec_evidence

_BODY = json.dumps({"nonce": "n", "state": "EXEC_FAILED", "error": "missing"})


def test_regular_receipt_is_read(tmp_path: Path) -> None:
    receipt = tmp_path / "r.exec.json"
    _ = receipt.write_text(_BODY)
    assert read_exec_evidence(receipt, "n") == "startup_failed"


def test_fifo_receipt_is_rejected_without_blocking(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fifo = tmp_path / "r.exec.json"
    os.mkfifo(fifo)
    results: list[str | None] = []
    reader = Thread(target=lambda: results.append(read_exec_evidence(fifo, "n")), daemon=True)
    reader.start()
    reader.join(timeout=2)
    if reader.is_alive():
        descriptor = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(descriptor)
        reader.join(timeout=2)
        pytest.fail("reading a FIFO receipt blocked")
    assert results == [None]
    assert "not a regular file" in caplog.text


def test_symlink_receipt_is_rejected(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    target = tmp_path / "target.json"
    _ = target.write_text(_BODY)
    link = tmp_path / "r.exec.json"
    link.symlink_to(target)
    assert read_exec_evidence(link, "n") is None
    assert "rejected" in caplog.text


def test_oversized_receipt_is_rejected(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    receipt = tmp_path / "r.exec.json"
    _ = receipt.write_text(_BODY + " " * 5000)
    assert read_exec_evidence(receipt, "n") is None
    assert "larger than" in caplog.text
