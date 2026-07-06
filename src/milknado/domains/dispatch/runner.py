"""Subprocess-worker dispatch: spawn, execute, and the sync/cancellable run paths.

Covers sync and detached headless spawning (piping the brief to stdin and
capturing combined output). The async worker, poll, and reconciliation
helpers live in async_run.py and reconcile.py respectively.
"""

from __future__ import annotations

import logging
import os
import secrets
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.common.agent_argv import _ALLOWED_WORKER_EXECUTABLES
from milknado.domains.dispatch._runstate import SUMMARY_TAIL_BYTES as _SUMMARY_TAIL_BYTES
from milknado.domains.dispatch._runstate import is_cancel_requested as _is_cancel_requested
from milknado.domains.dispatch._runstate import runs_dir as _runs_dir
from milknado.domains.dispatch._runstate import tail as _tail

_logger = logging.getLogger(__name__)

# Cooperative-cancel tuning for the async worker's poll loop: how often it
# checks the cancel sentinel / timeout, and how long it waits after SIGTERM
# before escalating to SIGKILL.
_CANCEL_POLL_SECS = 0.5
_CANCEL_GRACE_SECS = 5


def validate_worker_argv(argv: list[str]) -> None:
    """Reject a resolved worker argv whose executable isn't an allowed agent CLI.

    Guards every worker_cmd source — the explicit MCP arg and the resolved
    profile default — by checking the basename of argv[0] against the allowlist.
    """
    executable = Path(argv[0]).name if argv else ""
    if executable not in _ALLOWED_WORKER_EXECUTABLES:
        raise ValueError(
            "worker_cmd must start with one of "
            f"{sorted(_ALLOWED_WORKER_EXECUTABLES)!r}; got {executable!r}"
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


def _build_worker_env(extra: dict[str, str] | None = None) -> dict[str, str]:
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


def _spawn_worker(  # noqa: ANN001
    cwd: Path, node_id: int, log_fh, argv: list[str], run_id: str | None = None
) -> subprocess.Popen:
    """Popen a worker with the filtered env, stdin piped and output to log_fh.

    Shared by the blocking `_execute` (sync path) and the polling
    `_execute_cancellable` (async path) so the spawn + env + redirection contract
    lives in one place; only the wait strategy differs between the two. `cwd` is
    the working tree the worker runs in — the shared checkout under THIS_BRANCH,
    or the isolated worktree under ISOLATE. The worker receives MILKNADO_NODE_ID
    and, when this is a tracked run, MILKNADO_RUN_ID so it can deposit its result
    via milknado_deposit_result.
    """
    extra = {"MILKNADO_NODE_ID": str(node_id)}
    if run_id is not None:
        extra["MILKNADO_RUN_ID"] = run_id
    env = _build_worker_env(extra)
    return subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(cwd),
        env=env,
    )


def _execute(
    cwd: Path,
    node_id: int,
    log_path: Path,
    brief: str,
    argv: list[str],
    timeout: int,
    run_id: str | None = None,
) -> tuple[int, bool]:
    timed_out = False
    with log_path.open("wb") as log_fh:
        proc = _spawn_worker(cwd, node_id, log_fh, argv, run_id)
        try:
            proc.communicate(input=brief.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True
    exit_code = proc.returncode if proc.returncode is not None else -1
    return exit_code, timed_out


def _terminate_worker(proc: subprocess.Popen) -> None:
    """SIGTERM the worker, allow a grace window, then SIGKILL if still alive.

    The async subprocess shares the MCP server's process group, so signalling
    the group (`killpg`) would kill the server itself — we terminate this one
    process directly instead.
    """
    proc.terminate()
    try:
        proc.wait(timeout=_CANCEL_GRACE_SECS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _write_worker_stdin(stdin, brief: str) -> None:  # noqa: ANN001
    """Write the brief to the worker's stdin and close it.

    Runs in a daemon thread (async path) so a brief larger than the OS pipe
    buffer can't block before the cancel/timeout poll loop starts. A
    BrokenPipeError/OSError just means the worker already exited — nothing to do.
    """
    try:
        stdin.write(brief.encode("utf-8"))
        stdin.close()
    except (BrokenPipeError, OSError) as exc:
        _logger.debug(
            "Worker stdin unavailable while sending brief (worker likely exited): %s",
            exc,
        )


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
) -> tuple[int, bool, bool]:
    """Run a worker under a poll loop so a cancel sentinel can stop it mid-run.

    Mirrors `_execute`'s spawn but cannot use the blocking `communicate`: the
    brief is written to stdin and the loop polls `proc.poll()` on a short
    interval, enforcing the timeout deadline and checking the cancel sentinel.
    Returns `(exit_code, timed_out, cancelled)`. Used only by the async worker;
    the sync `_execute` path stays blocking.
    """
    timed_out = False
    cancelled = False
    deadline = time.monotonic() + timeout
    with log_path.open("wb") as log_fh:
        proc = _spawn_worker(cwd, node_id, log_fh, argv, run_id)
        # Write the brief from a daemon thread so a brief larger than the OS pipe
        # buffer can't block before the cancel/timeout poll loop starts — a
        # synchronous write would be unkillable until the worker drained stdin.
        if proc.stdin is not None:
            threading.Thread(
                target=_write_worker_stdin, args=(proc.stdin, brief), daemon=True
            ).start()
        while proc.poll() is None:
            if _is_cancel_requested(runs_dir, run_id):
                _terminate_worker(proc)
                cancelled = True
                break
            if time.monotonic() >= deadline:
                _terminate_worker(proc)
                timed_out = True
                break
            time.sleep(_CANCEL_POLL_SECS)
    exit_code = proc.returncode if proc.returncode is not None else -1
    return exit_code, timed_out, cancelled


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
) -> RunResult:
    argv = _resolve_worker_cmd(worker_cmd, default_cmd)
    log_path = (
        _runs_dir(project_root) / f"{run_id}.log"
        if run_id is not None
        else _log_path(project_root, node_id)
    )
    exit_code, timed_out = _execute(
        cwd or project_root, node_id, log_path, brief, argv, timeout_seconds, run_id
    )
    return RunResult(
        exit_code=exit_code,
        log_path=log_path,
        summary=_tail(log_path, _SUMMARY_TAIL_BYTES),
        timed_out=timed_out,
    )
