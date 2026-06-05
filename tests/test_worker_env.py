"""Tests for _build_worker_env: secrets stay in the parent, config passes through."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from milknado.domains.dispatch.runner import (
    _WORKER_ENV_ALLOWLIST,
    _build_worker_env,
    run_headless,
)


def test_allowlisted_system_vars_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/testuser")
    env = _build_worker_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"] == "/home/testuser"


def test_milknado_vars_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILKNADO_PROJECT_ROOT", "/tmp/proj")
    monkeypatch.setenv("MILKNADO_NODE_ID", "99")
    env = _build_worker_env()
    assert env["MILKNADO_PROJECT_ROOT"] == "/tmp/proj"
    assert env["MILKNADO_NODE_ID"] == "99"


def test_secrets_are_not_passed_to_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """API keys, tokens, and DB credentials must not reach worker subprocesses."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@host/db")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    env = _build_worker_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "DATABASE_URL" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GITHUB_TOKEN" not in env


def test_extra_vars_are_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _build_worker_env({"MILKNADO_NODE_ID": "42"})
    assert env["MILKNADO_NODE_ID"] == "42"


def test_extra_overrides_existing_milknado_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILKNADO_FOO", "original")
    env = _build_worker_env({"MILKNADO_FOO": "overridden"})
    assert env["MILKNADO_FOO"] == "overridden"


def test_none_extra_is_safe() -> None:
    env = _build_worker_env(None)
    assert isinstance(env, dict)


def test_allowlist_does_not_contain_credential_names() -> None:
    """Smoke-check: no obvious secret names snuck into the allowlist."""
    lower = {v.lower() for v in _WORKER_ENV_ALLOWLIST}
    for forbidden in ("key", "secret", "token", "password", "credential"):
        matches = [name for name in lower if forbidden in name]
        assert not matches, (
            f"allowlist contains a potentially sensitive name matching {forbidden!r}: {matches}"
        )


def test_run_headless_does_not_leak_planted_secret_to_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end on the dispatch path that DOES filter env (run_headless ->
    _execute -> _build_worker_env): a secret planted in the parent process must
    never appear in the spawned worker's environment. The worker dumps its own
    env to the log; we assert the secret value is absent and the allowlisted
    MILKNADO_NODE_ID injection is present."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "claude"
    # Passthrough agent stub: dumps the worker-visible environment to stdout (→ log).
    stub.write_text("#!/bin/sh\nexec env\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-PLANTED-LEAK-CANARY")

    result = run_headless(
        tmp_path, node_id=7, brief="hi", timeout_seconds=10, default_cmd="claude"
    )

    assert result.exit_code == 0
    log_text = result.log_path.read_text(encoding="utf-8")
    assert "sk-ant-PLANTED-LEAK-CANARY" not in log_text
    assert "ANTHROPIC_API_KEY" not in log_text
    assert "MILKNADO_NODE_ID=7" in log_text


def test_run_headless_brief_reaches_stdin_under_multiflag_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Brief must reach the worker via stdin even when default_cmd has multiple flags.

    Spec verification assumption: the todo-run path's default command becomes
    cfg.execution_agent (e.g. 'claude -p --dangerously-skip-permissions --allowedTools ...')
    after unification. The brief is piped to stdin; this test confirms that a
    multi-flag command preserves stdin-brief delivery.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "claude"
    # Stub echoes stdin to stdout (log), ignores flags.
    stub.write_text("#!/bin/sh\ncat\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    brief_marker = "BRIEF_MARKER_XYZ_12345"
    result = run_headless(
        tmp_path,
        node_id=1,
        brief=brief_marker,
        timeout_seconds=10,
        default_cmd="claude -p --dangerously-skip-permissions --allowedTools 'Read,Edit'",
    )

    assert result.exit_code == 0
    log_text = result.log_path.read_text(encoding="utf-8")
    assert brief_marker in log_text, (
        "brief must reach worker stdin under a multi-flag execution_agent command"
    )
