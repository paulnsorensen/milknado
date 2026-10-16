"""Liveness probes for tmux panes and executed-target receipts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from milknado.domains.dispatch import tmux_run
from milknado.domains.dispatch.tmux_run import read_exec_evidence


def test_group_owned_by_another_user_still_counts_as_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny(pgid: int, sig: int) -> None:
        _ = (pgid, sig)
        raise PermissionError

    monkeypatch.setattr(os, "killpg", deny)

    assert tmux_run._group_alive(1234) is True  # pyright: ignore[reportPrivateUsage]


def test_group_alive_is_false_for_an_absent_group(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(pgid: int, sig: int) -> None:
        _ = (pgid, sig)
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", absent)

    assert tmux_run._group_alive(1234) is False  # pyright: ignore[reportPrivateUsage]


def test_receipt_for_a_vanished_target_is_not_execution_evidence(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    assert process.wait(timeout=5) == 0
    receipt = tmp_path / "exec.json"
    body = {"nonce": "nonce", "state": "EXECUTED", "pid": process.pid, "start_token": 1.0}
    _ = receipt.write_text(json.dumps(body))

    assert read_exec_evidence(receipt, "nonce") is None
