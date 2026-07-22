"""Worker dispatch orchestration over the injected process port.

Builds worker commands and environments for sync and cancellable runs. Concrete
process launch, wait, timeout, and termination mechanics live in the adapter.
"""

from __future__ import annotations

import logging
import os
import secrets
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.common.agent_argv import (
    ALLOWED_WORKER_EXECUTABLES,
    POSITIONAL_BRIEF_EXECUTABLES,
)
from milknado.domains.dispatch._runstate import SUMMARY_TAIL_BYTES as _SUMMARY_TAIL_BYTES
from milknado.domains.dispatch._runstate import is_cancel_requested as _is_cancel_requested
from milknado.domains.dispatch._runstate import runs_dir as _runs_dir
from milknado.domains.dispatch._runstate import tail as _tail
from milknado.domains.dispatch.ports import ProcessPort

_logger = logging.getLogger(__name__)


def validate_worker_argv(argv: list[str]) -> None:
    """Reject a resolved worker argv whose executable isn't an allowed agent CLI.

    Guards every worker_cmd source — the explicit MCP arg and the resolved
    profile default — by checking the basename of argv[0] against the allowlist.
    """
    executable = Path(argv[0]).name if argv else ""
    if executable not in ALLOWED_WORKER_EXECUTABLES:
        raise ValueError(
            "worker_cmd must start with one of "
            f"{sorted(ALLOWED_WORKER_EXECUTABLES)!r}; got {executable!r}"
        )


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    log_path: Path
    summary: str
    timed_out: bool


@dataclass(frozen=True)
class AsyncStartRef:
    run_id: str
    log_path: Path


def _resolve_worker_cmd(explicit: str | None, default: str) -> list[str]:
    """Resolve the final worker argv from an explicit override or the profile default.

    explicit (MCP arg) → default (profile.execution_agent). Both are validated
    against the allowlist before use.
    """
    if explicit and explicit.strip():
        argv = shlex.split(explicit)
    else:
        argv = shlex.split(default)
    validate_worker_argv(argv)
    return argv


def _worker_invocation(argv: list[str], brief: str) -> tuple[tuple[str, ...], bytes]:
    if Path(argv[0]).name in POSITIONAL_BRIEF_EXECUTABLES:
        return (*argv, brief), b""
    return tuple(argv), brief.encode()


def _log_path(project_root: Path, node_id: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _runs_dir(project_root) / f"node-{node_id}-{stamp}-{secrets.token_hex(4)}.log"


_WORKER_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Runtime essentials
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        # Temporary files (platform-specific names)
        "TMPDIR",
        "TEMP",
        "TMP",
        # Locale / encoding
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "LC_COLLATE",
        "LC_MONETARY",
        "LC_NUMERIC",
        "LC_TIME",
        # Python runtime
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        # Git identity (not credentials)
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    }
)


def build_worker_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a filtered environment for worker subprocesses.

    Passes only allowlisted system vars plus MILKNADO_* config vars from the
    parent env. Secrets (API keys, tokens, DB URLs) stay in the parent only.
    """
    # INVARIANT: no MILKNADO_* var may hold a secret — every one is forwarded to
    # workers verbatim. Keep API keys, tokens, and DB URLs out of that namespace
    # (they belong to the parent only); a new MILKNADO_SECRET_* would leak here.
    env = {
        k: v
        for k, v in os.environ.items()
        if k in _WORKER_ENV_ALLOWLIST or k.startswith("MILKNADO_")
    }
    if extra:
        env.update(extra)
    return env


def _execute(
    project_root: Path,
    node_id: int,
    log_path: Path,
    brief: str,
    argv: list[str],
    timeout: int,
    run_id: str | None = None,
    *,
    canonical_root: Path | None = None,
    process: ProcessPort,
) -> tuple[int, bool]:
    root = canonical_root or project_root
    extra = {
        "MILKNADO_NODE_ID": str(node_id),
        "MILKNADO_PROJECT_ROOT": str(root.resolve()),
    }
    if run_id is not None:
        extra["MILKNADO_RUN_ID"] = run_id
    command, stdin = _worker_invocation(argv, brief)
    outcome = process.run(
        command,
        project_root,
        log_path,
        stdin,
        build_worker_env(extra),
        timeout,
    )
    return outcome.exit_code, outcome.timed_out


def _execute_cancellable(
    cwd: Path,
    node_id: int,
    log_path: Path,
    brief: str,
    argv: list[str],
    timeout: int,
    *,
    runs_dir: Path,
    run_id: str,
    project_root: Path,
    process: ProcessPort,
    on_started: Callable[[int], None] | None = None,
) -> tuple[int, bool, bool]:
    command, stdin = _worker_invocation(argv, brief)
    outcome = process.run(
        command,
        cwd,
        log_path,
        stdin,
        build_worker_env(
            {
                "MILKNADO_NODE_ID": str(node_id),
                "MILKNADO_RUN_ID": run_id,
                "MILKNADO_PROJECT_ROOT": str(project_root.resolve()),
            }
        ),
        timeout,
        cancel_requested=lambda: _is_cancel_requested(runs_dir, run_id),
        on_started=on_started,
    )
    return outcome.exit_code, outcome.timed_out, outcome.cancelled


def run_headless(
    project_root: Path,
    node_id: int,
    brief: str,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    *,
    run_id: str | None = None,
    default_cmd: str,
    cwd: Path | None = None,
    process: ProcessPort,
) -> RunResult:
    argv = _resolve_worker_cmd(worker_cmd, default_cmd)
    log_path = (
        _runs_dir(project_root) / f"{run_id}.log"
        if run_id is not None
        else _log_path(project_root, node_id)
    )
    worker_cwd = cwd or project_root
    _logger.info("worker dispatch started: run_id=%s node_id=%d", run_id, node_id)
    exit_code, timed_out = _execute(
        worker_cwd,
        node_id,
        log_path,
        brief,
        argv,
        timeout_seconds,
        run_id,
        canonical_root=project_root,
        process=process,
    )
    _logger.info(
        "worker dispatch terminal: run_id=%s node_id=%d exit_code=%d timed_out=%s",
        run_id,
        node_id,
        exit_code,
        timed_out,
    )
    return RunResult(
        exit_code=exit_code,
        log_path=log_path,
        summary=_tail(log_path, _SUMMARY_TAIL_BYTES),
        timed_out=timed_out,
    )
