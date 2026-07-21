from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from milknado.mcp._core import (
    RunDict,
    _worktree_main_checkout,
    build_run_dict,
    require_worker_run,
    resolve_project_root,
)


def test_run_dict_builder_is_the_canonical_superset() -> None:
    result = build_run_dict({"run_id": "run-1", "node_id": 3, "status": "running", "pid": 42})

    assert set(result) == set(RunDict.__annotations__)
    assert result == {
        "run_id": "run-1",
        "node_id": 3,
        "status": "running",
        "exit_code": None,
        "timed_out": None,
        "rebased": None,
        "pid": 42,
        "log_path": None,
        "summary": None,
        "result": None,
        "worktree_preserved": None,
        "error": None,
        "detail": None,
    }


def test_worker_cannot_select_different_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    injected = tmp_path / "owned"
    monkeypatch.setenv("MILKNADO_NODE_ID", "7")
    monkeypatch.setenv("MILKNADO_RUN_ID", "run-owned")
    monkeypatch.setenv("MILKNADO_PROJECT_ROOT", str(injected))

    with pytest.raises(ValueError, match="does not match injected root"):
        resolve_project_root(str(tmp_path / "other"))
    assert resolve_project_root(None) == injected.resolve()


def test_worker_identity_requires_injected_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MILKNADO_NODE_ID", "7")
    monkeypatch.delenv("MILKNADO_PROJECT_ROOT", raising=False)

    with pytest.raises(ValueError, match="requires MILKNADO_PROJECT_ROOT"):
        resolve_project_root(None)


def test_worker_cannot_deposit_for_different_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILKNADO_RUN_ID", "run-owned")

    require_worker_run("run-owned")
    with pytest.raises(ValueError, match="does not match injected run"):
        require_worker_run("run-other")


def test_git_root_discovery_warns_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failed = subprocess.CompletedProcess(
        args=["git"], returncode=128, stdout="", stderr="not a repository"
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: failed)

    with caplog.at_level(logging.WARNING, logger="milknado.mcp._core"):
        assert _worktree_main_checkout(tmp_path) is None

    assert caplog.messages == [
        f"git root discovery failed cwd={tmp_path} returncode=128 stderr=not a repository"
    ]
