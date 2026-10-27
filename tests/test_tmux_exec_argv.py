"""The tmux receipt executor must not import from the worker worktree cwd."""

from __future__ import annotations

import sys
from pathlib import Path

from milknado.adapters.tmux import _exec_receipt_command  # pyright: ignore[reportPrivateUsage]
from milknado.domains.dispatch.ports import RunWindow


def test_exec_receipt_command_isolates_the_interpreter_from_cwd(tmp_path: Path) -> None:
    window = RunWindow(
        run_id="r",
        argv=("echo", "hi"),
        cwd=tmp_path,
        log_path=tmp_path / "log",
        exit_code_path=tmp_path / "code",
        exec_receipt_path=tmp_path / "receipt",
        exec_nonce="nonce",
    )

    assert _exec_receipt_command(window)[:4] == (
        sys.executable,
        "-P",
        "-m",
        "milknado.loop._tmux_exec",
    )
