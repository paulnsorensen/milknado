from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Final

from milknado.domains.common.config import Gate
from milknado.loop import CompletionVerdict

_GATE_TIMEOUT_SECONDS: Final[float] = 1800.0
_GIT_QUERY_TIMEOUT_SECONDS: Final[float] = 60.0
_FEEDBACK_TAIL_CHARS: Final[int] = 2000

NO_GATES_CONFIGURED_MESSAGE: Final[str] = (
    "no quality_gates configured — set [milknado] quality_gates in milknado.toml "
    "(or a per-flavor [milknado.flavor.<flavor>] table; use [] to intentionally skip)"
)
_NO_CHANGE_FEEDBACK: Final[str] = (
    "you emitted the completion promise but produced no committed/stageable change"
)
_UNRESOLVABLE_BASE_FEEDBACK: Final[str] = (
    "the completion verifier has no immutable dispatch base, so it cannot confirm "
    "that committed work was produced"
)
_GIT_PROBE_FAILURE_FEEDBACK: Final[str] = (
    "the completion verifier could not inspect this worktree with git, so it cannot tell "
    "whether a change was produced; this is a harness failure, not a rejection of the work"
)
_logger = logging.getLogger(__name__)


class _GitProbeFailure(RuntimeError):
    """A git query the completion verifier depends on did not return an answer.

    Distinct from a clean tree: the verifier learned nothing either way, so the
    caller must not report the worker as having produced no change.
    """


def build_completion_verifier(
    worktree: Path,
    quality_gates: tuple[Gate, ...] | None,
    *,
    base_oid: str | None = None,
    in_place: bool = False,
    artifact_path: str | None = None,
) -> Callable[[], CompletionVerdict]:
    """Build the authoritative harness-side completion check.

    Isolated work is compared with the immutable commit captured at dispatch.
    In-place artifact tasks use artifact evidence only when gates are explicitly
    empty and an artifact path was declared.
    """

    def verify() -> CompletionVerdict:
        if quality_gates is None:
            return CompletionVerdict(ok=False, feedback=NO_GATES_CONFIGURED_MESSAGE)
        if not quality_gates:
            _logger.info("quality gates explicitly skipped for %s", worktree)
        gate_failure = _run_quality_gates(worktree, quality_gates)
        if gate_failure is not None:
            return CompletionVerdict(ok=False, feedback=gate_failure)
        if in_place and not quality_gates and artifact_path is not None:
            rejection = _artifact_rejection(worktree, artifact_path)
        else:
            rejection = _change_rejection(worktree, base_oid)
        return CompletionVerdict(ok=rejection is None, feedback=rejection or "")

    return verify


def _artifact_rejection(worktree: Path, artifact_path: str) -> str | None:
    artifact = Path(artifact_path)
    if artifact.is_absolute():
        return f"completion artifact path must be repo-relative: {artifact_path}"
    try:
        resolved_root = worktree.resolve()
        resolved_artifact = (resolved_root / artifact).resolve()
        resolved_artifact.relative_to(resolved_root)
    except (OSError, ValueError):
        return f"completion artifact path escapes worktree: {artifact_path}"
    try:
        if not resolved_artifact.is_file() or resolved_artifact.stat().st_size == 0:
            return f"completion artifact is missing or empty: {artifact_path}"
    except OSError as exc:
        return f"completion artifact could not be inspected: {artifact_path}: {exc}"
    return None


def _run_quality_gates(worktree: Path, gates: tuple[Gate, ...]) -> str | None:
    gate_env = {key: value for key, value in os.environ.items() if not key.startswith("MILKNADO_")}
    for gate in gates:
        try:
            result = subprocess.run(
                gate.command,
                cwd=worktree,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_GATE_TIMEOUT_SECONDS,
                env=gate_env,
            )
        except subprocess.TimeoutExpired:
            failure = f"quality gate timed out after {_GATE_TIMEOUT_SECONDS:g}s: {gate.command}"
            _logger.warning("%s", failure)
            return failure
        except OSError as exc:
            failure = f"quality gate could not start: {gate.command}: {exc}"
            _logger.warning("%s", failure)
            return failure
        output = f"{result.stdout}\n{result.stderr}"[-_FEEDBACK_TAIL_CHARS:]
        if result.returncode != 0:
            failure = f"quality gate failed (exit {result.returncode}): {gate.command}\n{output}"
            _logger.warning("%s", failure)
            return failure
        if gate.fail_on_stdout is not None and re.search(
            gate.fail_on_stdout, output, flags=re.MULTILINE
        ):
            failure = (
                f"quality gate output matched /{gate.fail_on_stdout}/: {gate.command}\n{output}"
            )
            _logger.warning("%s", failure)
            return failure
    return None


def _change_rejection(worktree: Path, base_oid: str | None) -> str | None:
    try:
        if _has_working_tree_change(worktree):
            return None
        if base_oid is None:
            _logger.warning("completion verifier: no immutable dispatch base for %s", worktree)
            return _UNRESOLVABLE_BASE_FEEDBACK
        if _has_committed_change(worktree, base_oid):
            return None
    except _GitProbeFailure as exc:
        _logger.warning("completion verifier: %s", exc)
        return f"{_GIT_PROBE_FAILURE_FEEDBACK}: {exc}"
    return _NO_CHANGE_FEEDBACK


def _has_working_tree_change(worktree: Path) -> bool:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_QUERY_TIMEOUT_SECONDS,
        ).stdout
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        UnicodeDecodeError,
    ) as exc:
        raise _GitProbeFailure(f"git status failed in {worktree}: {exc}") from exc
    return bool(status.strip())


def _has_committed_change(worktree: Path, base_oid: str) -> bool:
    try:
        diff = subprocess.run(
            ["git", "diff", "--quiet", base_oid, "HEAD"],
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=_GIT_QUERY_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError, UnicodeDecodeError) as exc:
        raise _GitProbeFailure(
            f"could not diff dispatch base {base_oid} in {worktree}: {exc}"
        ) from exc
    if diff.returncode == 1 and not diff.stderr:
        return True
    if diff.returncode != 0:
        raise _GitProbeFailure(
            f"git diff against dispatch base {base_oid} in {worktree} "
            f"exited {diff.returncode}: {diff.stderr.strip()}"
        )
    return False
