from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from milknado.app.project import (
    _worktree_main_checkout,  # pyright: ignore[reportPrivateUsage]
    require_worker_run,
    resolve_project_root,
)
from milknado.domains.common import BUILTIN_FLAVORS, NodeKind, NodeStatus
from milknado.mcp._core import (
    RunDict,
    _parse_flavor,  # pyright: ignore[reportPrivateUsage]
    _parse_kind,  # pyright: ignore[reportPrivateUsage]
    _parse_todo_status,  # pyright: ignore[reportPrivateUsage]
    build_run_dict,
)
from milknado.mcp.todo import milknado_todo_next
from milknado.mcp.todo_mutate import milknado_todo_set_status


def _call(tool: object, **kwargs: object) -> object:
    fn = cast(Callable[..., object], getattr(tool, "fn", tool))
    return fn(**kwargs)


def _failed_git_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["git"], returncode=128, stdout="", stderr="not a repository"
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


@pytest.mark.parametrize("value", [kind.value for kind in NodeKind])
def test_parse_kind_accepts_every_node_kind(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _parse_kind(value) is NodeKind(value)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("pending", NodeStatus.PENDING),
        ("in_progress", NodeStatus.RUNNING),
        ("blocked", NodeStatus.BLOCKED),
        ("done", NodeStatus.DONE),
    ],
)
def test_parse_todo_status_accepts_every_mcp_status(
    value: str, expected: NodeStatus, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _parse_todo_status(value) is expected
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_parse_todo_status_rejects_running_without_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(ValueError, match="invalid status 'running'"):
        _ = _parse_todo_status("running")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_parse_flavor_accepts_every_builtin_flavor_without_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = frozenset(BUILTIN_FLAVORS)
    for flavor in registry:
        assert _parse_flavor(flavor, registry) == flavor
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_parse_flavor_rejects_unknown_name_without_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(ValueError, match="invalid flavor 'not-a-flavor'"):
        _ = _parse_flavor("not-a-flavor", frozenset(BUILTIN_FLAVORS))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_invalid_mcp_handler_inputs_raise_without_stdout_or_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(ValueError, match="invalid kind 'not-a-kind'"):
        _ = _call(milknado_todo_next, kind="not-a-kind", project_root=str(tmp_path))
    with pytest.raises(ValueError, match="invalid status 'running'"):
        _ = _call(
            milknado_todo_set_status, node_id=1, status="running", project_root=str(tmp_path)
        )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_worker_cannot_select_different_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    injected = tmp_path / "owned"
    monkeypatch.setenv("MILKNADO_NODE_ID", "7")
    monkeypatch.setenv("MILKNADO_RUN_ID", "run-owned")
    monkeypatch.setenv("MILKNADO_PROJECT_ROOT", str(injected))

    with pytest.raises(ValueError, match="does not match injected root"):
        _ = resolve_project_root(str(tmp_path / "other"))
    assert resolve_project_root(None) == injected.resolve()


def test_worker_identity_requires_injected_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MILKNADO_NODE_ID", "7")
    monkeypatch.delenv("MILKNADO_PROJECT_ROOT", raising=False)

    with pytest.raises(ValueError, match="requires MILKNADO_PROJECT_ROOT"):
        _ = resolve_project_root(None)


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
    monkeypatch.setattr(subprocess, "run", _failed_git_run)

    with caplog.at_level(logging.WARNING, logger="milknado.app.project"):
        assert _worktree_main_checkout(tmp_path) is None

    assert caplog.messages == [
        f"git root discovery failed cwd={tmp_path} returncode=128 stderr=not a repository"
    ]
