"""Subprocess-worker dispatch plus the run-state / node-status reconciliation
helpers the MCP dispatch tools share.

Covers sync and detached headless spawning (piping the brief to stdin and
capturing combined output) alongside orphan-run recovery and the graph-side
`reconcile_node_status` transition.
"""

from __future__ import annotations

import json
import os
import secrets
import shlex
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.common import NodeStatus
from milknado.domains.dispatch._runstate import RUN_ID_RE as _RUN_ID_RE
from milknado.domains.dispatch._runstate import SUMMARY_TAIL_BYTES as _SUMMARY_TAIL_BYTES
from milknado.domains.dispatch._runstate import make_run_id as _make_run_id
from milknado.domains.dispatch._runstate import now_iso as _now_iso
from milknado.domains.dispatch._runstate import read_state as _read_state
from milknado.domains.dispatch._runstate import runs_dir as _runs_dir
from milknado.domains.dispatch._runstate import tail as _tail
from milknado.domains.dispatch._runstate import write_state as _write_state

_DEFAULT_WORKER_CMD = "claude -p"
# Grace beyond a run's own timeout before a still-"running" state file is
# treated as orphaned by a vanished worker thread (e.g. the server crashed).
_STALE_GRACE_SECONDS = 30


# Worker subprocesses may only invoke a known AI-agent CLI. Match on the bare
# executable name (basename of argv[0]) so neither a prefix trick
# (`claude-evil`) nor an absolute path (`/usr/bin/claude`) slips the check.
_ALLOWED_WORKER_EXECUTABLES: frozenset[str] = frozenset(
    {"claude", "codex", "cursor-agent", "gemini"}
)


def _validate_worker_argv(argv: list[str]) -> None:
    """Reject a resolved worker argv whose executable isn't an allowed agent CLI.

    Guards every worker_cmd source — the explicit MCP arg, the
    $MILKNADO_WORKER_CMD env fallback, and the built-in default — by checking
    the basename of argv[0] against the allowlist.
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
    state_path: Path


def _resolve_worker_cmd(explicit: str | None) -> list[str]:
    if explicit and explicit.strip():
        argv = shlex.split(explicit)
    else:
        env = os.environ.get("MILKNADO_WORKER_CMD", "").strip()
        argv = shlex.split(env) if env else shlex.split(_DEFAULT_WORKER_CMD)
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


def _execute(
    project_root: Path, node_id: int, log_path: Path, brief: str, argv: list[str], timeout: int
) -> tuple[int, bool]:
    timed_out = False
    env = _build_worker_env({"MILKNADO_NODE_ID": str(node_id)})
    with log_path.open("wb") as log_fh:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            env=env,
        )
        try:
            proc.communicate(input=brief.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True
    exit_code = proc.returncode if proc.returncode is not None else -1
    return exit_code, timed_out


def run_headless(
    project_root: Path,
    node_id: int,
    brief: str,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
) -> RunResult:
    argv = _resolve_worker_cmd(worker_cmd)
    log_path = _log_path(project_root, node_id)
    exit_code, timed_out = _execute(project_root, node_id, log_path, brief, argv, timeout_seconds)
    return RunResult(
        exit_code=exit_code,
        log_path=log_path,
        summary=_tail(log_path, _SUMMARY_TAIL_BYTES),
        timed_out=timed_out,
    )


def _async_worker(
    state_path: Path,
    project_root: Path,
    log_path: Path,
    brief: str,
    argv: list[str],
    timeout: int,
    base_state: dict,
) -> None:
    try:
        exit_code, timed_out = _execute(
            project_root, base_state["node_id"], log_path, brief, argv, timeout
        )
        terminal = "done" if exit_code == 0 and not timed_out else "failed"
        _write_state(
            state_path,
            {
                **base_state,
                "status": terminal,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "ended_at": _now_iso(),
            },
        )
    except Exception as exc:
        # Spawn failures (FileNotFoundError on bad worker_cmd, OSError, etc.) or
        # write failures must not leave the state file stuck on "running" — that
        # would silently lock out the node forever. Append to the log so any
        # partial subprocess output that did make it to disk is preserved.
        try:
            with log_path.open("a", encoding="utf-8") as log_fh:
                log_fh.write(f"\n--- async worker raised: {type(exc).__name__}: {exc} ---\n")
        except OSError:
            # Annotating the log is best-effort; the terminal state write below
            # is what actually releases the node, so a log-write failure here is
            # deliberately ignored.
            pass
        _write_state(
            state_path,
            {
                **base_state,
                "status": "failed",
                "exit_code": -1,
                "timed_out": False,
                "ended_at": _now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


def start_headless_async(
    project_root: Path,
    node_id: int,
    brief: str,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    run_id: str | None = None,
) -> AsyncStartRef:
    argv = _resolve_worker_cmd(worker_cmd)
    # The caller (milknado_todo_run_start) claims the node under a run_id before
    # spawning, then hands that same id here so the node row and the run-state file
    # agree on the fence; standalone callers let us mint one.
    run_id = run_id or _make_run_id(node_id)
    runs_dir = _runs_dir(project_root)
    log_path = runs_dir / f"{run_id}.log"
    state_path = runs_dir / f"{run_id}.state.json"
    base_state = {
        "run_id": run_id,
        "node_id": node_id,
        "started_at": _now_iso(),
        "log_path": str(log_path),
        "timeout_seconds": timeout_seconds,
    }
    _write_state(
        state_path,
        {
            **base_state,
            "status": "running",
            "exit_code": None,
            "timed_out": False,
            "ended_at": None,
        },
    )
    threading.Thread(
        target=_async_worker,
        args=(state_path, project_root, log_path, brief, argv, timeout_seconds, base_state),
        daemon=True,
    ).start()
    return AsyncStartRef(run_id=run_id, log_path=log_path, state_path=state_path)


def poll_async_run(project_root: Path, run_id: str) -> dict:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    state_path = _runs_dir(project_root) / f"{run_id}.state.json"
    if not state_path.exists():
        raise ValueError(f"run {run_id!r} not found")
    state = _read_state(state_path)
    log_path = Path(state["log_path"])
    state["summary"] = _tail(log_path, _SUMMARY_TAIL_BYTES)
    return state


def fail_stale_running_runs(project_root: Path, node_id: int) -> list[dict]:
    """Flip this node's orphaned "running" state files to "failed".

    The async worker runs in a daemon thread; if the server process dies
    mid-run the thread dies with it and never writes a terminal state, leaving
    the state file stuck on "running" and the node locked on RUNNING forever
    (orphan reconciliation only recovers runs that reached done/failed). A run
    still "running" past its own timeout plus a grace margin can only mean the
    thread vanished — `_execute` kills and reaps the subprocess at timeout, so
    a live worker always reaches a terminal write. Mark such runs failed so the
    caller's reconciliation can release the node. Returns the runs it flipped.
    """
    now = datetime.now(UTC)
    flipped: list[dict] = []
    for state_path in _runs_dir(project_root).glob(f"node-{node_id}-*.state.json"):
        try:
            state = _read_state(state_path)
        except (OSError, json.JSONDecodeError):
            continue
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
        failed = {
            **state,
            "status": "failed",
            "exit_code": -1,
            "timed_out": False,
            "ended_at": _now_iso(),
            "error": "worker vanished before writing terminal state (stale running run)",
        }
        _write_state(state_path, failed)
        flipped.append(failed)
    return flipped


def find_terminal_runs_for_node(
    project_root: Path, node_id: int, run_id: str | None = None
) -> list[dict]:
    """Scan the runs dir for state files whose node matches and which have reached
    a terminal status. Used by callers that want to reconcile orphaned runs
    (started, worker finished, but never polled — node still marked RUNNING).

    When `run_id` is given, filter to that fence: only the node's current owner's
    terminal file is returned, so a stale terminal file from an older run cannot be
    selected. Callers must filter to the fence BEFORE picking the latest run —
    otherwise a stale run with a later `ended_at` could mask the owner's file."""
    runs_dir = _runs_dir(project_root)
    out: list[dict] = []
    for state_path in runs_dir.glob(f"node-{node_id}-*.state.json"):
        try:
            state = _read_state(state_path)
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("node_id") != node_id or state.get("status") not in ("done", "failed"):
            continue
        if run_id is not None and state.get("run_id") != run_id:
            continue
        out.append(state)
    return out


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
