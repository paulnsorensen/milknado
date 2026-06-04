"""Subprocess-worker dispatch plus the run-state / node-status reconciliation
helpers the MCP dispatch tools share.

Covers sync and detached headless spawning (piping the brief to stdin and
capturing combined output) alongside orphan-run recovery and the graph-side
`reconcile_node_status` transition.
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

from milknado.domains.common import NodeStatus, pid_alive
from milknado.domains.dispatch._runstate import RUN_ID_RE as _RUN_ID_RE
from milknado.domains.dispatch._runstate import SUMMARY_TAIL_BYTES as _SUMMARY_TAIL_BYTES
from milknado.domains.dispatch._runstate import clear_cancel as _clear_cancel
from milknado.domains.dispatch._runstate import is_cancel_requested as _is_cancel_requested
from milknado.domains.dispatch._runstate import make_run_id as _make_run_id
from milknado.domains.dispatch._runstate import now_iso as _now_iso
from milknado.domains.dispatch._runstate import runs_dir as _runs_dir
from milknado.domains.dispatch._runstate import tail as _tail

_logger = logging.getLogger(__name__)

# Grace beyond a run's own timeout before a still-"running" state file is
# treated as orphaned by a vanished worker thread (e.g. the server crashed).
_STALE_GRACE_SECONDS = 30
# Cooperative-cancel tuning for the async worker's poll loop: how often it
# checks the cancel sentinel / timeout, and how long it waits after SIGTERM
# before escalating to SIGKILL.
_CANCEL_POLL_SECS = 0.5
_CANCEL_GRACE_SECS = 5


def _open_worker_graph(project_root: Path):  # noqa: ANN202
    """Open a fresh MikadoGraph for the worker thread.

    sqlite connections are not cross-thread, so the async worker cannot reuse the
    dispatching tool's connection — it opens its own in-thread for the run-row
    writes. Imported lazily to avoid a domains -> _mcp_core import cycle (the same
    pattern the detached _ralph_node_runner uses).
    """
    from milknado._mcp_core import open_graph

    graph, _cfg = open_graph(project_root)
    return graph


# Worker subprocesses may only invoke a known AI-agent CLI. Match on the bare
# executable name (basename of argv[0]) so neither a prefix trick
# (`claude-evil`) nor an absolute path (`/usr/bin/claude`) slips the check.
_ALLOWED_WORKER_EXECUTABLES: frozenset[str] = frozenset(
    {"claude", "codex", "cursor-agent", "gemini"}
)


def _validate_worker_argv(argv: list[str]) -> None:
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
    _validate_worker_argv(argv)
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
    project_root: Path, node_id: int, log_fh, argv: list[str], run_id: str | None = None
) -> subprocess.Popen:
    """Popen a worker with the filtered env, stdin piped and output to log_fh.

    Shared by the blocking `_execute` (sync path) and the polling
    `_execute_cancellable` (async path) so the spawn + env + redirection contract
    lives in one place; only the wait strategy differs between the two. The worker
    receives MILKNADO_NODE_ID and, when this is a tracked run, MILKNADO_RUN_ID so
    it can deposit its result via milknado_deposit_result.
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
        cwd=str(project_root),
        env=env,
    )


def _execute(
    project_root: Path,
    node_id: int,
    log_path: Path,
    brief: str,
    argv: list[str],
    timeout: int,
    run_id: str | None = None,
) -> tuple[int, bool]:
    timed_out = False
    with log_path.open("wb") as log_fh:
        proc = _spawn_worker(project_root, node_id, log_fh, argv, run_id)
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
    project_root: Path,
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
        proc = _spawn_worker(project_root, node_id, log_fh, argv, run_id)
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
) -> RunResult:
    argv = _resolve_worker_cmd(worker_cmd, default_cmd)
    log_path = (
        _runs_dir(project_root) / f"{run_id}.log"
        if run_id is not None
        else _log_path(project_root, node_id)
    )
    exit_code, timed_out = _execute(
        project_root, node_id, log_path, brief, argv, timeout_seconds, run_id
    )
    return RunResult(
        exit_code=exit_code,
        log_path=log_path,
        summary=_tail(log_path, _SUMMARY_TAIL_BYTES),
        timed_out=timed_out,
    )


def _async_worker(
    project_root: Path,
    log_path: Path,
    brief: str,
    argv: list[str],
    timeout: int,
    base_state: dict,
) -> None:
    rdir = _runs_dir(project_root)
    run_id = base_state["run_id"]
    # sqlite connections are not cross-thread: open a fresh in-thread graph for
    # the terminal write rather than reusing the dispatching tool's connection.
    graph = _open_worker_graph(project_root)
    try:
        exit_code, timed_out, cancelled = _execute_cancellable(
            project_root,
            base_state["node_id"],
            log_path,
            brief,
            argv,
            timeout,
            runs_dir=rdir,
            run_id=run_id,
        )
        # The worker owns the terminal write for a cancelled run: run_cancel only
        # requests cancellation, so finalizing here (not there) is what closes the
        # state-clobber race the old signal-then-overwrite path left open.
        if cancelled:
            graph.finish_run(
                run_id,
                status="failed",
                exit_code=-1,
                timed_out=timed_out,
                ended_at=_now_iso(),
                error="cancelled",
            )
        else:
            terminal = "done" if exit_code == 0 and not timed_out else "failed"
            graph.finish_run(
                run_id,
                status=terminal,
                exit_code=exit_code,
                timed_out=timed_out,
                ended_at=_now_iso(),
            )
    except Exception as exc:
        _logger.warning("async worker for run %s raised %s: %s", run_id, type(exc).__name__, exc)
        # Spawn failures (FileNotFoundError on bad worker_cmd, OSError, etc.) or
        # write failures must not leave the run stuck on "running" — that would
        # silently lock out the node forever. Append to the log so any partial
        # subprocess output that did make it to disk is preserved.
        try:
            with log_path.open("a", encoding="utf-8") as log_fh:
                log_fh.write(f"\n--- async worker raised: {type(exc).__name__}: {exc} ---\n")
        except OSError:
            # Annotating the log is best-effort; the terminal run write below is
            # what actually releases the node, so a log-write failure here is
            # deliberately ignored.
            pass
        graph.finish_run(
            run_id,
            status="failed",
            exit_code=-1,
            timed_out=False,
            ended_at=_now_iso(),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        graph.close()
        # Clear the sentinel after the terminal write (both branches) so a reused
        # run dir never carries a stale cancel request into the next run.
        _clear_cancel(rdir, run_id)


def start_headless_async(
    project_root: Path,
    node_id: int,
    brief: str,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    run_id: str | None = None,
    *,
    default_cmd: str,
) -> AsyncStartRef:
    argv = _resolve_worker_cmd(worker_cmd, default_cmd)
    # The caller (milknado_todo_run_start) claims the node under a run_id before
    # spawning, then hands that same id here so the node row and the run row agree
    # on the fence; standalone callers let us mint one.
    run_id = run_id or _make_run_id(node_id)
    runs_dir = _runs_dir(project_root)
    log_path = runs_dir / f"{run_id}.log"
    started_at = _now_iso()
    base_state = {"run_id": run_id, "node_id": node_id}
    # Insert the rescuable "running" run row BEFORE spawning the worker thread, on
    # a short-lived connection (the thread opens its own for the terminal write).
    graph = _open_worker_graph(project_root)
    try:
        graph.start_run(run_id, node_id, str(log_path), started_at, timeout_seconds)
    finally:
        graph.close()
    threading.Thread(
        target=_async_worker,
        args=(project_root, log_path, brief, argv, timeout_seconds, base_state),
        daemon=True,
    ).start()
    return AsyncStartRef(run_id=run_id, log_path=log_path)


def poll_async_run(graph, project_root: Path, run_id: str) -> dict:  # noqa: ANN001
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    state = graph.get_run(run_id)
    if state is None:
        raise ValueError(f"run {run_id!r} not found")
    # Derive the log path from the validated run_id rather than trusting the
    # stored field: no arbitrary-file read via a tampered log_path.
    log_path = _runs_dir(project_root) / f"{run_id}.log"
    state["log_path"] = str(log_path)
    state["summary"] = _tail(log_path, _SUMMARY_TAIL_BYTES)
    return state


def fail_stale_running_runs(graph, node_id: int) -> list[dict]:  # noqa: ANN001
    """Flip this node's orphaned "running" runs to "failed".

    The async worker runs in a daemon thread; if the server process dies
    mid-run the thread dies with it and never writes a terminal state, leaving
    the run stuck on "running" and the node locked on RUNNING forever
    (orphan reconciliation only recovers runs that reached done/failed). For the
    worker model a run still "running" past its own timeout plus a grace margin
    can only mean the thread vanished — `_execute` kills and reaps the subprocess
    at timeout, so a live worker always reaches a terminal write. A detached-ralph
    runner has no such killer (its only timeout is `wait_for_next_completion`), so
    this also skips any run whose recorded pid is still alive — slow, not dead.
    Mark the rest failed so the caller's reconciliation can release the node.
    Returns the runs it flipped.
    """
    now = datetime.now(UTC)
    flipped: list[dict] = []
    for state in graph.runs_for_node(node_id):
        if state.get("status") != "running":
            continue
        started_at = state.get("started_at")
        timeout = state.get("timeout_seconds")
        if not started_at or timeout is None:
            continue
        try:
            started = datetime.fromisoformat(started_at)
        except ValueError:
            continue
        if (now - started).total_seconds() <= timeout + _STALE_GRACE_SECONDS:
            continue
        # A detached-ralph runner records its own pid; a live process past the
        # grace window is slow (e.g. wedged in `git rebase`, which has no
        # subprocess timeout), not orphaned. Flipping it would thrash a run about
        # to write "done" — and could cross-fail another run kind sharing the
        # node. The async-worker path records no pid, so its orphan sweep (server
        # crash → daemon thread vanished, no terminal write) is unchanged:
        # pid-unknown still falls through to the timeout flip.
        pid = state.get("pid")
        if pid is not None and pid_alive(pid):
            # Log the skip: a run wedged past timeout+grace that keeps getting
            # skipped as "slow, not dead" would otherwise leave no trace, since
            # the caller discards this function's return value.
            _logger.info(
                "stale-run sweep skipped live run %s (node %s, pid %s) — slow, not orphaned",
                state.get("run_id"),
                node_id,
                pid,
            )
            continue
        # A pid-recording detached run that reached here has a dead pid (the
        # liveness skip above did not fire), so "vanished" understates it — name
        # the exited process. The async-worker path records no pid and keeps the
        # daemon-thread "vanished" wording.
        error = (
            f"detached runner (pid {pid}) exited before writing terminal state (stale running run)"
            if pid is not None
            else "worker vanished before writing terminal state (stale running run)"
        )
        ended_at = _now_iso()
        graph.finish_run(
            state["run_id"],
            status="failed",
            exit_code=-1,
            timed_out=False,
            ended_at=ended_at,
            error=error,
        )
        flipped.append(
            {
                **state,
                "status": "failed",
                "exit_code": -1,
                "timed_out": False,
                "ended_at": ended_at,
                "error": error,
            }
        )
    return flipped


def find_terminal_runs_for_node(
    graph,
    node_id: int,
    run_id: str | None = None,  # noqa: ANN001
) -> list[dict]:
    """Return this node's runs that reached a terminal status. Used by callers that
    want to reconcile orphaned runs (started, worker finished, but never polled —
    node still marked RUNNING).

    When `run_id` is given, filter to that fence: only the node's current owner's
    terminal run is returned, so a stale terminal run from an older run cannot be
    selected. Callers must filter to the fence BEFORE picking the latest run —
    otherwise a stale run with a later `ended_at` could mask the owner's run."""
    return graph.runs_for_node(node_id, terminal_only=True, run_id=run_id)


def latest_terminal_run(runs: list[dict]) -> dict | None:
    """Return the most recent terminal run by ended_at, or None if runs is empty.

    Deterministic selection for reconcile: when a node has multiple terminal
    runs (e.g. a prior run finished after the node was reset and re-run), the
    latest ended_at wins. Callers that fence on run_id must filter to the fence
    first (see find_terminal_runs_for_node) so a stale run with a later ended_at
    cannot mask the current owner's terminal file. Shared by the headless
    (mcp_run) and detached-ralph (mcp_ralph) dispatch paths.
    """
    terminal = [r for r in runs if r.get("status") in ("done", "failed")]
    if not terminal:
        return None
    return max(terminal, key=lambda r: r.get("ended_at") or "")


def reconcile_node_status(graph, node_id: int, run_status: str, run_id: str | None = None) -> None:  # noqa: ANN001
    """Transition a node to its run's terminal status, idempotently and fenced.

    When `run_id` is given AND the node carries a run_id, the write goes through
    `graph.mark_terminal`, whose atomic `WHERE run_id = ? AND status = 'running'`
    is the real fence: a node re-claimed under a newer run_id (or already terminal)
    matches zero rows and is left for its current owner. This closes the TOCTOU a
    pre-check-then-unfenced-write left open — two concurrent reconcilers can both
    pass a Python-level run_id check, but only the matching atomic UPDATE lands.

    A node with no run_id (in-process TUI / legacy) has no fence to honour, so it
    falls back to the unconditional transition; only a RUNNING node is touched, so
    a node reset out of RUNNING is never force-marked. Shared by the sync (`mcp_run`)
    and detached worktree (`mcp_ralph`) dispatch tools.
    """
    node = graph.get_node(node_id)
    if node is None or node.status != NodeStatus.RUNNING:
        return
    target = {"done": NodeStatus.DONE, "failed": NodeStatus.FAILED}.get(run_status)
    if target is None:
        return
    if run_id is not None and node.run_id is not None:
        graph.mark_terminal(node_id, run_id, target)
    elif target is NodeStatus.DONE:
        graph.mark_done(node_id)
    else:
        graph.mark_failed(node_id)


def reconcile_orphan_node(graph, node_id: int) -> None:  # noqa: ANN001
    """Flip stale 'running' runs to failed, reconcile the node's status to the
    latest terminal run. No-op if the node is not RUNNING.

    Shared entry-point for mcp_run (#39) and mcp_ralph (#54) — the three-call
    orphan-release pattern must not be duplicated inline in either caller.
    """
    fail_stale_running_runs(graph, node_id)
    winner = latest_terminal_run(find_terminal_runs_for_node(graph, node_id))
    if winner is not None:
        reconcile_node_status(graph, node_id, winner["status"])
