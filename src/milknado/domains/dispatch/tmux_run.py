"""Tmux run-window policy: fail-closed preflight, the run-inline pane waiter,
window lifecycle reconciliation, and attach preconditions.

The tmux target is derived deterministically from the run_id at read time —
no ``runs``-table column. Pane liveness is additive only (attach precondition,
diagnostics); pid-liveness stays the sole authority for graph-state
transitions.
"""

from __future__ import annotations

import logging
import os
import signal
import stat
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import msgspec

from milknado.domains.common import group_state, identity_state, pid_alive
from milknado.domains.dispatch._runstate import (
    RUN_ID_RE,
    is_cancel_requested,
)
from milknado.domains.dispatch._tmux_receipt import (
    ExecutedReceipt,
    FailedReceipt,
    FinishedReceipt,
    Receipt,
)
from milknado.domains.dispatch.ports import RunWindow, TmuxPort

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)
_RECEIPT_LIMIT_BYTES = 4096

_PANE_POLL_SECS = 0.5
_PANE_KILL_GRACE_SECS = 5.0


def ensure_tmux_ready(tmux: TmuxPort) -> None:
    """Fail-closed preflight for an explicitly tmux-requested dispatch.

    tmux was asked for, so a missing binary or an unstartable server fails the
    dispatch loudly — never a silent fallback to the detached path.
    """
    if not tmux.available():
        raise ValueError(
            "tmux was requested but the tmux binary is not on PATH; "
            + "install tmux or dispatch without use_tmux"
        )
    try:
        tmux.ensure_session()
    except Exception as exc:
        raise ValueError(f"tmux server could not start: {exc}") from exc


def _group_alive(pgid: int) -> bool:
    return group_state(pgid) != "gone"


def _terminate_pane(pane_pid: int) -> None:
    """SIGTERM the pane's process group, allow a grace window, then SIGKILL.

    The pane process is its own group leader, so group-signalling cannot touch
    the MCP server. The pane shell dies on SIGTERM while its protected-exec
    wrapper still confirms worker cleanup, so the wait covers the whole group.
    The window is left alone: remain-on-exit keeps the dead pane around,
    honouring failed-run window preservation.
    """
    with suppress(ProcessLookupError):
        os.killpg(pane_pid, signal.SIGTERM)
    deadline = time.monotonic() + _PANE_KILL_GRACE_SECS
    while _group_alive(pane_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _group_alive(pane_pid):
        with suppress(ProcessLookupError):
            os.killpg(pane_pid, signal.SIGKILL)


def read_exit_code(path: Path) -> int:
    """Exit code the window wrapper recorded; -1 when missing or garbled
    (pane killed before the runner could finish)."""
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return -1


def _live_target(pid: int, start_token: float) -> bool:
    return identity_state(pid, start_token) == "live"


def _read_receipt_bytes(path: Path) -> bytes | None:
    """Read a bounded receipt without following symlinks or blocking on special files."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        _logger.warning("exec receipt %s rejected: %s", path, exc)
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            _logger.warning("exec receipt %s rejected: not a regular file", path)
            return None
        data = os.read(descriptor, _RECEIPT_LIMIT_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(data) > _RECEIPT_LIMIT_BYTES:
        _logger.warning(
            "exec receipt %s rejected: larger than %d bytes", path, _RECEIPT_LIMIT_BYTES
        )
        return None
    return data


def read_exec_evidence(path: Path, nonce: str) -> Literal["executed", "startup_failed"] | None:
    data = _read_receipt_bytes(path)
    if data is None:
        return None
    try:
        receipt = cast(Receipt, msgspec.json.decode(data, type=Receipt))
    except msgspec.DecodeError as exc:
        _logger.warning("exec receipt %s rejected: %s", path, exc)
        return None
    if receipt.nonce != nonce:
        return None
    if isinstance(receipt, FailedReceipt):
        return "startup_failed"
    if isinstance(receipt, FinishedReceipt):
        return "executed" if receipt.returncode >= 0 else None
    if isinstance(receipt, ExecutedReceipt) and receipt.start_token is not None:
        return "executed" if _live_target(receipt.pid, receipt.start_token) else None
    return None


def execute_in_window(
    tmux: TmuxPort,
    window: RunWindow,
    timeout: int,
    rdir: Path,
    *,
    on_start: Callable[[int], None] | None = None,
) -> tuple[int, bool, bool]:
    """Wait for the pane and persist its PID; protected launch writes execution evidence.

    Returns ``(exit_code, timed_out, cancelled)``.
    """
    pane_pid = tmux.open_run_window(window)
    if on_start is not None:
        on_start(pane_pid)
    timed_out = False
    cancelled = False
    deadline = time.monotonic() + timeout
    while pid_alive(pane_pid):
        if is_cancel_requested(rdir, window.run_id):
            _terminate_pane(pane_pid)
            cancelled = True
            break
        if time.monotonic() >= deadline:
            _terminate_pane(pane_pid)
            timed_out = True
            break
        time.sleep(_PANE_POLL_SECS)
    return read_exit_code(window.exit_code_path), timed_out, cancelled


def cleanup_run_window(tmux: TmuxPort, run_state: Mapping[str, object]) -> bool:
    """Per-row window reconciliation — the runs table is the expected set.

    A run that completed successfully has its window cleaned up (normally the
    window wrapper already did; this is the backstop after e.g. a server
    restart). A failed run's window is preserved for inspection; a running
    run's window is left alone. One exact-match query per run row — never a
    pattern sweep of the session's windows. Returns True when a window was
    killed.
    """
    if run_state.get("status") != "done":
        return False
    if not tmux.available():
        return False
    run_id = run_state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return False
    return tmux.kill_window(run_id)


def reconcile_run_window(tmux: TmuxPort, run_state: Mapping[str, object]) -> None:
    """Best-effort poll-time reconcile hook: tmux trouble must never break a poll."""
    try:
        _ = cleanup_run_window(tmux, run_state)
    except Exception as exc:
        _logger.warning(
            "tmux window reconcile failed: run_id=%s error=%s",
            run_state.get("run_id"),
            exc,
            exc_info=True,
        )


def resolve_attach_target(graph: MikadoGraph, tmux: TmuxPort, run_id: str) -> str:
    """Attach preconditions, each failure mode with its own message.

    Pane existence is inherently a tmux question, so this is the one place
    window-liveness gates behaviour — it never feeds node-status transitions.
    """
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    state = graph.runs.get(run_id)
    if state is None:
        raise ValueError(f"run {run_id!r} not found")
    if state.get("status") != "running":
        raise ValueError(
            f"run {run_id!r} already finished (status={state.get('status')}); "
            + "a failed run's window, if any, is preserved — inspect it with tmux directly"
        )
    if not tmux.available():
        raise ValueError("tmux is not installed; cannot attach")
    if not tmux.window_exists(run_id):
        raise ValueError(
            f"run {run_id!r} has no tmux window — it was dispatched without use_tmux, "
            + "or its window is gone"
        )
    return tmux.target_for(run_id)
