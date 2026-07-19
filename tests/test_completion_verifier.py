from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from milknado.domains.execution.completion import build_completion_verifier


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_committed_change_is_compared_to_immutable_dispatch_base(tmp_path: Path) -> None:
    with patch("milknado.domains.execution.completion.subprocess.run") as run:
        run.side_effect = [_completed(stdout=""), _completed(returncode=1)]

        verdict = build_completion_verifier(tmp_path, (), base_oid="dispatch-oid")()

    assert verdict.ok is True
    assert run.call_args_list[1].args[0] == [
        "git",
        "diff",
        "--quiet",
        "dispatch-oid",
        "HEAD",
    ]
    assert run.call_args_list[1].kwargs == {
        "cwd": tmp_path,
        "capture_output": True,
        "text": True,
        "timeout": 60.0,
    }


def test_missing_dispatch_base_fails_closed(tmp_path: Path) -> None:
    with patch(
        "milknado.domains.execution.completion.subprocess.run",
        return_value=_completed(stdout=""),
    ):
        verdict = build_completion_verifier(tmp_path, (), base_oid=None)()

    assert verdict.ok is False
    assert "no immutable dispatch base" in verdict.feedback


def test_git_status_error_fails_closed(tmp_path: Path) -> None:
    failed = MagicMock(side_effect=OSError("git unavailable"))
    with patch("milknado.domains.execution.completion.subprocess.run", failed):
        verdict = build_completion_verifier(tmp_path, (), base_oid="dispatch-oid")()

    assert verdict.ok is False
    assert verdict.feedback == (
        "you emitted the completion promise but produced no committed/stageable change"
    )
    assert failed.call_count == 2


def test_nonzero_git_diff_error_is_not_a_change(tmp_path: Path) -> None:
    with patch("milknado.domains.execution.completion.subprocess.run") as run:
        run.side_effect = [_completed(stdout=""), _completed(returncode=128)]
        verdict = build_completion_verifier(tmp_path, (), base_oid="bad-oid")()

    assert verdict.ok is False
    assert verdict.feedback == (
        "you emitted the completion promise but produced no committed/stageable change"
    )
