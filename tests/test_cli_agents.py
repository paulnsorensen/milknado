"""Tests for `milknado agents check`.

WHY: the command's output is meant to be pasteable into a shared log or PR.
`build_planning_subprocess` injects the full subprocess env (a copy of
`os.environ`, minus external-MCP keys) under the `env` extra, so any secret in
the shell — API tokens, registry creds — must never reach stdout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from milknado.cli import app
from milknado.cli_agents import _redact_extra

runner = CliRunner()


def test_agents_check_does_not_leak_env_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Secret-shaped var with no "MCP" in the name → build_minimal_mcp_env keeps it
    # in the subprocess env, so the display path is the only thing standing between
    # it and stdout.
    secret = "s3cr3t-do-not-print-this-value"
    monkeypatch.setenv("FAKE_PROVIDER_TOKEN", secret)
    # Stop rich from soft-wrapping the extras dict across lines so the substring
    # check sees the value contiguously if it leaks.
    monkeypatch.setenv("COLUMNS", "100000")

    result = runner.invoke(app, ["agents", "check", "--project-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert secret not in result.output
    # The env injection is still signalled (summarised), not silently dropped.
    assert "env" in result.output


def test_redact_extra_summarises_env_to_count() -> None:
    # The env dict is replaced by a value-free count so secrets never render.
    out = _redact_extra("env", {"A": "1", "SECRET_TOKEN": "hunter2", "C": "3"})
    assert out == "<3 env vars>"
    assert "hunter2" not in str(out)


def test_redact_extra_masks_input_as_stdin() -> None:
    assert _redact_extra("input", "# sample planning context\n") == "<stdin>"


def test_redact_extra_passes_through_non_secret_scalars() -> None:
    # Keys like `text` carry no secrets and must survive unchanged.
    assert _redact_extra("text", True) is True


def test_redact_extra_does_not_summarise_non_dict_env() -> None:
    # The dict guard means a non-dict `env` is left alone rather than mislabelled
    # — the count format only applies to the real environment mapping.
    assert _redact_extra("env", "not-a-dict") == "not-a-dict"
