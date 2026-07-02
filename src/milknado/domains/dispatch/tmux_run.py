"""Tmux run-window policy: fail-closed preflight, the run-once pane waiter,
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
import time
from contextlib import suppress
from pathlib import Path

from milknado.adapters.tmux import RunWindow, TmuxAdapter, TmuxDispatchError
from milknado.domains.common import pid_alive
from milknado.domains.dispatch._runstate import (
    RUN_ID_RE,
    is_cancel_requested,
)

_logger = logging.getLogger(__name__)

_PANE_POLL_SECS = 0.5
_PANE_KILL_GRACE_SECS = 5.0


def ensure_tmux_ready(tmux: TmuxAdapter) -> None:
    """Fail-closed preflight for an explicitly tmux-requested dispatch.

    tmux was asked for, so a missing binary or an unstartable server fails the
    dispatch loudly — never a silent fallback to the detached path.
    """
    if not tmux.available():
        raise ValueError(
            "tmux was requested but the tmux binary is not on PATH; "
            "install tmux or dispatch without use_tmux"
        )
    try:
        tmux.ensure_session()
    except TmuxDispatchError as exc:
        raise ValueError(f"tmux was requested but its server could not start: {exc}") from exc


def _terminate_pane(pane_pid: int) -> None:
    """SIGTERM the pane's process group, allow a grace window, then SIGKILL.

    The pane process is its own group leader, so group-signalling cannot touch
    the MCP server. The window is left alone: remain-on-exit keeps the dead
    pane around, honouring failed-run window preservation.
    """
    with suppress(ProcessLookupError):
        os.killpg(pane_pid, signal.SIGTERM)
    deadline = time.monotonic() + _PANE_KILL_GRACE_SECS
    while pid_alive(pane_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if pid_alive(pane_pid):
        with suppress(ProcessLookupError):
            os.killpg(pane_pid, signal.SIGKILL)


def read_exit_code(path: Path) -> int:
    """Exit code the window wrapper recorded; -1 when missing or garbled
    (pane killed before the runner could finish)."""
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return -1


def execute_in_window(
    tmux: TmuxAdapter, window: RunWindow, timeout: int, rdir: Path
) -> tuple[int, bool, bool]:
    """Run a run-once worker inside a tmux window; mirrors ``_execute_cancellable``.

    The waiter polls the pane pid (there is no Popen handle — the process is a
    child of the tmux server) and enforces the same cancel-sentinel and
    timeout contract. Returns ``(exit_code, timed_out, cancelled)``.
    """
    pane_pid = tmux.open_run_window(window)
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


def cleanup_run_window(tmux: TmuxAdapter, run_state: dict) -> bool:
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
    if not run_id or not tmux.window_exists(run_id):
        return False
    tmux.kill_window(run_id)
    return True


def reconcile_run_window(project_root: Path, run_state: dict) -> None:
    """Best-effort poll-time reconcile hook: tmux trouble must never break a poll."""
    try:
        cleanup_run_window(TmuxAdapter(project_root), run_state)
    except Exception as exc:  # noqa: BLE001 — diagnostics-only path; poll result is the payload
        _logger.warning(
            "tmux window reconcile failed for run %s: %s", run_state.get("run_id"), exc
        )


def resolve_attach_target(graph, tmux: TmuxAdapter, run_id: str) -> str:  # noqa: ANN001
    """Attach preconditions, each failure mode with its own message.

    Pane existence is inherently a tmux question, so this is the one place
    window-liveness gates behaviour — it never feeds node-status transitions.
    """
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    state = graph.get_run(run_id)
    if state is None:
        raise ValueError(f"run {run_id!r} not found")
    if state.get("status") != "running":
        raise ValueError(
            f"run {run_id!r} already finished (status={state.get('status')}); "
            "a failed run's window, if any, is preserved — inspect it with tmux directly"
        )
    if not tmux.available():
        raise ValueError("tmux is not installed; cannot attach")
    if not tmux.window_exists(run_id):
        raise ValueError(
            f"run {run_id!r} has no tmux window — it was dispatched without use_tmux, "
            "or its window is gone"
        )
    return tmux.target_for(run_id)
