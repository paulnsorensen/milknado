"""Subprocess-worker dispatch plus the run-state / node-status reconciliation
helpers the MCP dispatch tools share.

Covers sync and detached headless spawning (piping the brief to stdin and
capturing combined output) alongside orphan-run recovery and the graph-side
`reconcile_node_status` transition.
"""

from __future__ import annotations

import json
import os
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
        return shlex.split(explicit)
    env = os.environ.get("MILKNADO_WORKER_CMD", "").strip()
    if env:
        return shlex.split(env)
    return shlex.split(_DEFAULT_WORKER_CMD)


def _log_path(project_root: Path, node_id: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _runs_dir(project_root) / f"node-{node_id}-{stamp}.log"


def _execute(
    project_root: Path, node_id: int, log_path: Path, brief: str, argv: list[str], timeout: int
) -> tuple[int, bool]:
    timed_out = False
    # Make env explicit so the worker can attribute follow-up nodes it tracks
    # back to the task it is running (milknado_track_follow_up reads this).
    env = {**os.environ, "MILKNADO_NODE_ID": str(node_id)}
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
) -> AsyncStartRef:
    argv = _resolve_worker_cmd(worker_cmd)
    run_id = _make_run_id(node_id)
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


def find_terminal_runs_for_node(project_root: Path, node_id: int) -> list[dict]:
    """Scan the runs dir for state files whose node matches and which have reached
    a terminal status. Used by callers that want to reconcile orphaned runs
    (started, worker finished, but never polled — node still marked RUNNING)."""
    runs_dir = _runs_dir(project_root)
    out: list[dict] = []
    for state_path in runs_dir.glob(f"node-{node_id}-*.state.json"):
        try:
            state = _read_state(state_path)
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("node_id") == node_id and state.get("status") in ("done", "failed"):
            out.append(state)
    return out


def reconcile_node_status(graph, node_id: int, run_status: str) -> None:  # noqa: ANN001
    """Transition a node to its run's terminal status, idempotently.

    Only a RUNNING node can validly transition to a terminal status. Any other
    state (already terminal, or reset externally) is left alone so reconciliation
    never raises InvalidTransition — the first reconcile of a run wins, and a node
    taken out of RUNNING is not force-marked. Shared by the sync (`mcp_run`) and
    detached worktree (`mcp_ralph`) dispatch tools.
    """
    node = graph.get_node(node_id)
    if node is None or node.status != NodeStatus.RUNNING:
        return
    if run_status == "done":
        graph.mark_done(node_id)
    elif run_status == "failed":
        graph.mark_failed(node_id)
