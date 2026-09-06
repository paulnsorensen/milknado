"""Stable subprocess launchers for worker tests."""

from __future__ import annotations

import os
import shlex
from collections.abc import Callable
from pathlib import Path

import pytest

from milknado.domains.dispatch import validate_worker_argv

_PASSTHROUGH = '#!/bin/sh\nexec "$@"\n'


def install_worker_command(
    bindir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    agent: str,
    script: str,
) -> str:
    """Expose a stable shell symlink and a non-executable worker script."""
    # Bare agent names satisfy the worker allowlist.
    validate_worker_argv([agent])
    bindir.mkdir(parents=True, exist_ok=True)
    script_path = bindir / f"{agent}-worker.sh"
    _ = script_path.write_text(script, encoding="utf-8")
    # Non-executable script data avoids measured macOS loader stalls.
    script_path.chmod(0o644)
    (bindir / agent).symlink_to("/bin/sh")
    path = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    monkeypatch.setenv("PATH", path)
    return f"{agent} {shlex.quote(str(script_path))}"


def install_worker_stub(
    bindir: Path, monkeypatch: pytest.MonkeyPatch, *, agent: str = "claude"
) -> Callable[[str], str]:
    """Return a worker command builder that preserves each command suffix."""
    base = install_worker_command(bindir, monkeypatch, agent=agent, script=_PASSTHROUGH)

    def _command(suffix: str) -> str:
        return f"{base} {suffix}" if suffix else base

    return _command
