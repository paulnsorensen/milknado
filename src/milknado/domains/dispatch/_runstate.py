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
            fh.seek(size - max_bytes)
        data = fh.read()
    return data.decode("utf-8", errors="replace")


def exit_code_path(runs_dir: Path, run_id: str) -> Path:
    """Where a tmux window wrapper records the runner's exit code."""
    return runs_dir / f"{run_id}.rc"


def brief_path(runs_dir: Path, run_id: str) -> Path:
    """Where the run-inline tmux path stages the worker brief (stdin redirect)."""
    return runs_dir / f"{run_id}.brief"


def cancel_path(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / f"{run_id}.cancel"


def request_cancel(runs_dir: Path, run_id: str) -> None:
    """Atomically create the cancel sentinel for a run.

    The async worker polls `is_cancel_requested` and terminates its own
    subprocess when the sentinel appears — the cooperative-cancel signal that
    replaces process-group signalling for runs that share the MCP server's group.
    Write-tmp-then-replace so a concurrent poll never sees a
    half-written sentinel.
    """
    path = cancel_path(runs_dir, run_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("", encoding="utf-8")
    tmp.replace(path)


def is_cancel_requested(runs_dir: Path, run_id: str) -> bool:
    return cancel_path(runs_dir, run_id).exists()


def clear_cancel(runs_dir: Path, run_id: str) -> None:
    """Best-effort removal of the cancel sentinel after a terminal write so a
    reused run dir cannot carry a stale request into the next run."""
    try:
        cancel_path(runs_dir, run_id).unlink()
    except FileNotFoundError:
        pass  # sentinel already absent — nothing to clear
