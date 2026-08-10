from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from milknado.domains.execution.completion import build_completion_verifier


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


_NO_CHANGE = "you emitted the completion promise but produced no committed/stageable change"
_HARNESS_FAILURE = "this is a harness failure, not a rejection of the work"


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


def test_git_status_error_is_not_reported_as_the_worker_producing_no_change(
    tmp_path: Path,
) -> None:
    """A git tooling failure and a clean tree are different outcomes.

    The verifier still fails closed — it cannot confirm work was produced — but
    the feedback goes verbatim into the worker's next prompt, so blaming the
    worker for the harness's broken git sends it to redo work that was fine.
    """
    failed = MagicMock(side_effect=OSError("git unavailable"))
    with patch("milknado.domains.execution.completion.subprocess.run", failed):
        verdict = build_completion_verifier(tmp_path, (), base_oid="dispatch-oid")()

    assert verdict.ok is False
    assert verdict.feedback != _NO_CHANGE
    assert _HARNESS_FAILURE in verdict.feedback
    assert "git unavailable" in verdict.feedback, "feedback must name the underlying git error"
    assert failed.call_count == 1, "a failed git status must not be followed by a doomed git diff"


def test_nonzero_git_diff_error_is_not_reported_as_no_change(tmp_path: Path) -> None:
    with patch("milknado.domains.execution.completion.subprocess.run") as run:
        run.side_effect = [
            _completed(stdout=""),
            _completed(returncode=128, stderr="fatal: bad object bad-oid"),
        ]
        verdict = build_completion_verifier(tmp_path, (), base_oid="bad-oid")()

    assert verdict.ok is False
    assert verdict.feedback != _NO_CHANGE
    assert _HARNESS_FAILURE in verdict.feedback
    assert "exited 128" in verdict.feedback
    assert "fatal: bad object bad-oid" in verdict.feedback


def test_git_diff_raising_is_not_reported_as_no_change(tmp_path: Path) -> None:
    """The committed-change probe carries the same contract as the status probe:
    git raising (timeout, missing binary, undecodable output) means the verifier
    got no answer, which is not the same as a clean tree."""
    with patch("milknado.domains.execution.completion.subprocess.run") as run:
        run.side_effect = [
            _completed(stdout=""),
            subprocess.TimeoutExpired("git diff", 60.0),
        ]
        verdict = build_completion_verifier(tmp_path, (), base_oid="dispatch-oid")()

    assert verdict.ok is False
    assert verdict.feedback != _NO_CHANGE
    assert _HARNESS_FAILURE in verdict.feedback
    assert "dispatch-oid" in verdict.feedback, "feedback must name the base it could not diff"


def test_clean_tree_still_rejects_the_worker(tmp_path: Path) -> None:
    """The other side of the split: git answered, and the answer was 'no change'."""
    with patch("milknado.domains.execution.completion.subprocess.run") as run:
        run.side_effect = [_completed(stdout=""), _completed(returncode=0)]
        verdict = build_completion_verifier(tmp_path, (), base_oid="dispatch-oid")()

    assert verdict.ok is False
    assert verdict.feedback == _NO_CHANGE
