"""Spawn a subprocess worker, pipe the brief to stdin, capture combined output."""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_DEFAULT_WORKER_CMD = "claude -p"
_SUMMARY_TAIL_BYTES = 2000
_RUN_ID_RE = re.compile(r"^node-\d+-\d{8}T\d{6}Z-[0-9a-f]+$")


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


def _runs_dir(project_root: Path) -> Path:
    runs_dir = project_root / ".milknado" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _log_path(project_root: Path, node_id: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _runs_dir(project_root) / f"node-{node_id}-{stamp}.log"


def _make_run_id(node_id: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"node-{node_id}-{stamp}-{secrets.token_hex(2)}"


def _tail(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        data = fh.read()
    return data.decode("utf-8", errors="replace")


def _write_state(state_path: Path, payload: dict) -> None:
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(state_path)


def _read_state(state_path: Path) -> dict:
    return json.loads(state_path.read_text(encoding="utf-8"))


def _execute(
    project_root: Path, log_path: Path, brief: str, argv: list[str], timeout: int
) -> tuple[int, bool]:
    timed_out = False
    with log_path.open("wb") as log_fh:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
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
    exit_code, timed_out = _execute(project_root, log_path, brief, argv, timeout_seconds)
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
        exit_code, timed_out = _execute(project_root, log_path, brief, argv, timeout)
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
