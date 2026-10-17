"""Shared helpers that drive the real `milknado run` CLI in a subprocess."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

RUN_TIMEOUT_SECONDS = 60.0


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_cli(repo: Path) -> tuple[int, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    argv = [sys.executable, "-c", "from milknado.cli import app; app()"]
    proc = subprocess.Popen(
        [*argv, "run", "--project-root", str(repo)],
        cwd=repo,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=RUN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        pytest.fail(f"milknado run did not exit on its own; output={output[-3000:]!r}")
    return proc.returncode, output


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()
