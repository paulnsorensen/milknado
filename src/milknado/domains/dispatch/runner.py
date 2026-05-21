"""Spawn a subprocess worker, pipe the brief to stdin, capture combined output."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_DEFAULT_WORKER_CMD = "claude -p"
_SUMMARY_TAIL_BYTES = 2000


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    log_path: Path
    summary: str
    timed_out: bool


def _resolve_worker_cmd(explicit: str | None) -> list[str]:
    if explicit and explicit.strip():
        return shlex.split(explicit)
    env = os.environ.get("MILKNADO_WORKER_CMD", "").strip()
    if env:
        return shlex.split(env)
    return shlex.split(_DEFAULT_WORKER_CMD)


def _log_path(project_root: Path, node_id: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = project_root / ".milknado" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / f"node-{node_id}-{stamp}.log"


def _tail(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        data = fh.read()
    return data.decode("utf-8", errors="replace")


def run_headless(
    project_root: Path,
    node_id: int,
    brief: str,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
) -> RunResult:
    argv = _resolve_worker_cmd(worker_cmd)
    log_path = _log_path(project_root, node_id)

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
            proc.communicate(input=brief.encode("utf-8"), timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True

    exit_code = proc.returncode if proc.returncode is not None else -1
    return RunResult(
        exit_code=exit_code,
        log_path=log_path,
        summary=_tail(log_path, _SUMMARY_TAIL_BYTES),
        timed_out=timed_out,
    )
