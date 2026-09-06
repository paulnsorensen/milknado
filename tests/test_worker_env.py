"""Tests for build_worker_env: secrets stay in the parent, config passes through."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from milknado.adapters import ProcessAdapter
from milknado.domains.dispatch import run_headless
from milknado.domains.dispatch.runner import (
    _WORKER_ENV_ALLOWLIST,  # pyright: ignore[reportPrivateUsage]
    build_worker_env,
)
from tests.worker_fixtures import install_worker_command


def test_allowlisted_system_vars_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/testuser")
    env = build_worker_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"] == "/home/testuser"


def test_milknado_vars_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILKNADO_PROJECT_ROOT", "/tmp/proj")
    monkeypatch.setenv("MILKNADO_NODE_ID", "99")
    env = build_worker_env()
    assert env["MILKNADO_PROJECT_ROOT"] == "/tmp/proj"
    assert env["MILKNADO_NODE_ID"] == "99"


def test_secrets_are_not_passed_to_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """API keys, tokens, and DB credentials must not reach worker subprocesses."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@host/db")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    env = build_worker_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "DATABASE_URL" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GITHUB_TOKEN" not in env


def test_extra_vars_are_merged() -> None:
    env = build_worker_env({"MILKNADO_NODE_ID": "42"})
    assert env["MILKNADO_NODE_ID"] == "42"


def test_extra_overrides_existing_milknado_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILKNADO_FOO", "original")
    env = build_worker_env({"MILKNADO_FOO": "overridden"})
    assert env["MILKNADO_FOO"] == "overridden"


def test_none_extra_is_safe() -> None:
    env = build_worker_env(None)
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
    _execute -> build_worker_env): a secret planted in the parent process must
    never appear in the spawned worker's environment. The worker dumps its own
    env to the log; we assert the secret value is absent and the allowlisted
    MILKNADO_NODE_ID injection is present."""
    default_cmd = install_worker_command(
        tmp_path / "bin", monkeypatch, agent="claude", script="#!/bin/sh\nexec env\n"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-PLANTED-LEAK-CANARY")

    result = run_headless(
        tmp_path,
        node_id=7,
        brief="hi",
        timeout_seconds=10,
        default_cmd=default_cmd,
        process=ProcessAdapter(),
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
    default_cmd = (
        install_worker_command(
            tmp_path / "bin",
            monkeypatch,
            agent="claude",
            script="#!/bin/sh\ncat\n",
        )
        + " -p --dangerously-skip-permissions --allowedTools 'Read,Edit'"
    )

    brief_marker = "BRIEF_MARKER_XYZ_12345"
    result = run_headless(
        tmp_path,
        node_id=1,
        brief=brief_marker,
        timeout_seconds=10,
        default_cmd=default_cmd,
        process=ProcessAdapter(),
    )

    assert result.exit_code == 0
    log_text = result.log_path.read_text(encoding="utf-8")
    assert brief_marker in log_text, (
        "brief must reach worker stdin under a multi-flag execution_agent command"
    )


def test_run_headless_delivers_omp_brief_as_positional_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_cmd = (
        install_worker_command(
            tmp_path / "bin",
            monkeypatch,
            agent="omp",
            script='#!/bin/sh\nprintf "argv=<%s>\\n" "$*"\nprintf "stdin=<%s>\\n" "$(cat)"\n',
        )
        + " -p --auto-approve --no-session"
    )

    brief_marker = "OMP_BRIEF_MARKER_XYZ_12345"
    result = run_headless(
        tmp_path,
        node_id=1,
        brief=brief_marker,
        timeout_seconds=10,
        default_cmd=default_cmd,
        process=ProcessAdapter(),
    )

    assert result.exit_code == 0
    log_text = result.log_path.read_text(encoding="utf-8")
    assert f"argv=<-p --auto-approve --no-session {brief_marker}>" in log_text
    assert "stdin=<>" in log_text


def test_run_headless_forwards_openrouter_key_to_omp_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An omp worker authenticates via OPENROUTER_API_KEY, so it must reach the
    omp subprocess even though build_worker_env filters secrets by default."""
    default_cmd = (
        install_worker_command(
            tmp_path / "bin",
            monkeypatch,
            agent="omp",
            script="#!/bin/sh\nexec env\n",
        )
        + " -p --auto-approve --no-session"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-CANARY-12345")

    result = run_headless(
        tmp_path,
        node_id=1,
        brief="hi",
        timeout_seconds=10,
        default_cmd=default_cmd,
        process=ProcessAdapter(),
    )

    assert result.exit_code == 0
    log_text = result.log_path.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=or-key-CANARY-12345" in log_text


def test_run_headless_does_not_forward_openrouter_key_to_non_omp_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-omp worker (e.g. claude) has no use for OPENROUTER_API_KEY, so the
    default secret-filtering behavior of build_worker_env must still apply."""
    default_cmd = (
        install_worker_command(
            tmp_path / "bin",
            monkeypatch,
            agent="claude",
            script="#!/bin/sh\nexec env\n",
        )
        + " -p"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-CANARY-12345")

    result = run_headless(
        tmp_path,
        node_id=1,
        brief="hi",
        timeout_seconds=10,
        default_cmd=default_cmd,
        process=ProcessAdapter(),
    )

    assert result.exit_code == 0
    log_text = result.log_path.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY" not in log_text


def test_worker_launcher_rejects_unknown_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="worker_cmd must start with"):
        _ = install_worker_command(
            tmp_path / "bin", monkeypatch, agent="not-an-agent", script="#!/bin/sh\n"
        )


def test_worker_launcher_preserves_subprocess_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = '#!/bin/sh\nprintf "argv=<%s>\\n" "$*"\nprintf "stdin=<%s>\\n" "$(cat)"\nexit 7\n'
    default_cmd = (
        install_worker_command(
            tmp_path / "bin",
            monkeypatch,
            agent="claude",
            script=script,
        )
        + " first 'two words'"
    )

    result = run_headless(
        tmp_path,
        node_id=1,
        brief="brief-marker",
        timeout_seconds=10,
        default_cmd=default_cmd,
        process=ProcessAdapter(),
    )

    script_path = Path(shlex.split(default_cmd)[1])
    assert script_path.stat().st_mode & 0o111 == 0
    assert result.exit_code == 7
    assert result.log_path.read_text(encoding="utf-8") == (
        "argv=<first two words>\nstdin=<brief-marker>\n"
    )
