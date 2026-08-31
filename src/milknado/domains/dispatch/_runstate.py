"""Shared run-id, log-tail, and cancel-sentinel helpers for `.milknado/runs/` —
used by both the headless worker path (`runner.py`) and the worktree-isolated
ralph path (`mcp_ralph.py` + `_ralph_node_runner.py`). Run ids share one namespace
and format so reconciliation is uniform regardless of which path spawned the run.
Run *state* lives in the SQLite `runs` table; only log files and cancel sentinels
remain on the filesystem here.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime
from pathlib import Path

SUMMARY_TAIL_BYTES = 2000
RUN_ID_RE = re.compile(r"^node-\d+-\d{8}T\d{6}Z-[0-9a-f]+$")


def runs_dir(project_root: Path) -> Path:
    d = project_root / ".milknado" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_run_id(node_id: int) -> str:
    # 4 bytes of entropy (not 2): back-to-back runs of the same node land in the
    # same wall-clock second, so the suffix is the only thing distinguishing
    # their ids.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"node-{node_id}-{stamp}-{secrets.token_hex(4)}"


def tail(path: Path, max_bytes: int = SUMMARY_TAIL_BYTES) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            _ = fh.seek(size - max_bytes)
        data = fh.read()
    return data.decode("utf-8", errors="replace")


def tail_latest_iteration_log(log_dir: Path, max_bytes: int = SUMMARY_TAIL_BYTES) -> str:
    """Tail the most recent per-iteration file in a `.ralph-logs` directory.

    In-process (executor-owned) ralph runs write one file per iteration
    under ``log_dir`` (see ``loop/_agent.py``'s ``_new_output_sink``) rather
    than a single flat log a detached (subprocess) run's stdout redirects
    to — there is nothing at the conventional ``<run_id>.log`` path for
    these runs. Iteration filenames are zero-padded-iteration-prefixed, so
    sorting by name gives chronological order; the newest file is the run's
    current activity.
    """
    if not log_dir.is_dir():
        return ""
    files = sorted(log_dir.glob("*.log"))
    if not files:
        return ""
    return tail(files[-1], max_bytes)


def exit_code_path(runs_dir: Path, run_id: str) -> Path:
    """Where a tmux window wrapper records the runner's exit code."""
    return runs_dir / f"{run_id}.rc"


def brief_path(runs_dir: Path, run_id: str) -> Path:
    """Where the run-inline tmux path stages the worker brief (stdin redirect)."""
    return runs_dir / f"{run_id}.brief"


def cancel_path(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / f"{run_id}.cancel"


def request_cancel(runs_dir: Path, run_id: str) -> None:
    cancel_path(runs_dir, run_id).touch()


def is_cancel_requested(runs_dir: Path, run_id: str) -> bool:
    return cancel_path(runs_dir, run_id).exists()


def clear_cancel(runs_dir: Path, run_id: str) -> None:
    cancel_path(runs_dir, run_id).unlink(missing_ok=True)
