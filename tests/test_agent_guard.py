"""The session PATH guard blocks every allowlisted agent CLI."""

from __future__ import annotations

import subprocess

import pytest

from milknado.domains.common.agent_argv import ALLOWED_WORKER_EXECUTABLES
from tests.worker_fixtures import GUARD_MESSAGE


@pytest.mark.parametrize("name", sorted(ALLOWED_WORKER_EXECUTABLES))
def test_bare_agent_name_hits_the_guard(name: str) -> None:
    result = subprocess.run([name, "--version"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert GUARD_MESSAGE in result.stderr


def test_guard_survives_per_test_path_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-test PATH edit that is undone must still leave the guard in force."""
    with monkeypatch.context() as scoped:
        scoped.setenv("PATH", "/nonexistent")
    result = subprocess.run(["claude", "--version"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert GUARD_MESSAGE in result.stderr
