"""Detached-headless async dispatch: background thread worker, start, and poll."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from milknado.domains.dispatch._runstate import (
    RUN_ID_RE as _RUN_ID_RE,
)
from milknado.domains.dispatch._runstate import (
    SUMMARY_TAIL_BYTES as _SUMMARY_TAIL_BYTES,
)
from milknado.domains.dispatch._runstate import (
    clear_cancel as _clear_cancel,
)
from milknado.domains.dispatch._runstate import (
    make_run_id as _make_run_id,
)
from milknado.domains.dispatch._runstate import (
    now_iso as _now_iso,
)
from milknado.domains.dispatch._runstate import (
    runs_dir as _runs_dir,
)
from milknado.domains.dispatch._runstate import (
    tail as _tail,
)
from milknado.domains.dispatch.runner import (
    AsyncStartRef,
    _execute_cancellable,
    _resolve_worker_cmd,
)

_logger = logging.getLogger(__name__)


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
    # The caller (milknado_run_once_start) claims the node under a run_id before
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
