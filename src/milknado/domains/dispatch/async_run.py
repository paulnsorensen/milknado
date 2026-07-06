"""Detached-headless async dispatch: background thread worker, start, and poll."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from pathlib import Path

from milknado.adapters.tmux import RunWindow, TmuxAdapter
from milknado.domains.dispatch._runstate import (
    RUN_ID_RE as _RUN_ID_RE,
)
from milknado.domains.dispatch._runstate import (
    SUMMARY_TAIL_BYTES as _SUMMARY_TAIL_BYTES,
)
from milknado.domains.dispatch._runstate import (
    brief_path as _brief_path,
)
from milknado.domains.dispatch._runstate import (
    clear_cancel as _clear_cancel,
)
from milknado.domains.dispatch._runstate import (
    exit_code_path as _exit_code_path,
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
from milknado.domains.dispatch.isolate import (
    IsolateContext,
    MergeBackResult,
    merge_back_isolated,
)
from milknado.domains.dispatch.runner import (
    AsyncStartRef,
    _build_worker_env,
    _execute_cancellable,
    _resolve_worker_cmd,
)
from milknado.domains.dispatch.tmux_run import execute_in_window as _execute_in_window

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


def _run_worker_process(
    cwd: Path,
    log_path: Path,
    brief: str,
    argv: list[str],
    *,
    timeout: int,
    base_state: dict,
    rdir: Path,
    tmux: TmuxAdapter | None,
    graph,  # noqa: ANN001
) -> tuple[int, bool, bool]:
    """Execute the worker on the requested substrate: tmux window or subprocess.

    `cwd` is the working tree the worker runs in — the shared checkout under
    THIS_BRANCH, or the isolated worktree under ISOLATE. Both paths honour the
    same contract — brief on stdin, combined output in the run log, cancel-
    sentinel + timeout enforcement — and return ``(exit_code, timed_out,
    cancelled)``.
    """
    if tmux is None:
        return _execute_cancellable(
            cwd,
            base_state["node_id"],
            log_path,
            brief,
            argv,
            timeout,
            runs_dir=rdir,
            run_id=base_state["run_id"],
        )
    return _run_in_tmux_window(
        cwd,
        log_path,
        brief,
        argv,
        timeout=timeout,
        base_state=base_state,
        rdir=rdir,
        tmux=tmux,
        graph=graph,
    )


def _run_in_tmux_window(
    cwd: Path,
    log_path: Path,
    brief: str,
    argv: list[str],
    *,
    timeout: int,
    base_state: dict,
    rdir: Path,
    tmux: TmuxAdapter,
    graph,  # noqa: ANN001
) -> tuple[int, bool, bool]:
    """Stage the brief and run the worker inside its named tmux window.

    No stdin pipe into a tmux pane: the brief is staged to a file the window
    wrapper redirects into the worker. `cwd` is the worker's working tree.
    """
    run_id = base_state["run_id"]
    staged_brief = _brief_path(rdir, run_id)
    staged_brief.write_text(brief, encoding="utf-8")
    log_path.touch()
    window = RunWindow(
        run_id=run_id,
        argv=tuple(argv),
        cwd=cwd,
        log_path=log_path,
        exit_code_path=_exit_code_path(rdir, run_id),
        env=_build_worker_env(
            {"MILKNADO_NODE_ID": str(base_state["node_id"]), "MILKNADO_RUN_ID": run_id}
        ),
        brief_path=staged_brief,
    )
    # Persist the pane pid on the run row (fenced on status='running'): a tmux
    # pane outlives an MCP-server restart, so the stale sweep's pid-liveness
    # skip and milknado_run_cancel's group-kill must be able to see it —
    # unlike the in-process subprocess path, which dies with the server.
    return _execute_in_window(
        tmux,
        window,
        timeout,
        rdir,
        on_start=lambda pane_pid: graph.set_run_pid(run_id, pane_pid),
    )


def _async_merge_back(
    project_root: Path, merge_ctx: IsolateContext | None, terminal: str
) -> MergeBackResult | None:
    """Rebase-merge an ISOLATE branch back after a clean worker exit.

    Returns the ``MergeBackResult`` when a merge-back was requested, or None when
    there was nothing to merge (THIS_BRANCH, merge_back=False, or a non-``done``
    worker) so the run row's ``rebased`` stays None — matching the old shared-
    checkout dispatch that never rebased. A preserved (un-torn-down) worktree is
    both logged and carried on the result so the caller can persist it for poll.
    """
    if merge_ctx is None or terminal != "done":
        return None
    result = merge_back_isolated(project_root, merge_ctx)
    if result.worktree_preserved is not None:
        _logger.warning(
            "ISOLATE merge-back for branch %s did not tear down; preserved worktree %s",
            merge_ctx.worker_branch,
            result.worktree_preserved,
        )
    return result


def _async_worker(
    project_root: Path,
    log_path: Path,
    brief: str,
    argv: list[str],
    timeout: int,
    base_state: dict,
    tmux: TmuxAdapter | None = None,
    *,
    cwd: Path | None = None,
    merge_ctx: IsolateContext | None = None,
) -> None:
    rdir = _runs_dir(project_root)
    run_id = base_state["run_id"]
    # sqlite connections are not cross-thread: open a fresh in-thread graph for
    # the terminal write rather than reusing the dispatching tool's connection.
    graph = _open_worker_graph(project_root)
    try:
        exit_code, timed_out, cancelled = _run_worker_process(
            cwd or project_root,
            log_path,
            brief,
            argv,
            timeout=timeout,
            base_state=base_state,
            rdir=rdir,
            tmux=tmux,
            graph=graph,
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
            merge = _async_merge_back(project_root, merge_ctx, terminal)
            if merge is not None and merge.rebased is False:
                # ISOLATE merge_back requested but the branch did not land: the
                # deliverable never reached the dispatch branch, so this is a real
                # failure — do not reconcile the node DONE off a clean worker exit.
                terminal = "failed"
            # Persist a preserved (un-torn-down) worktree via the run row's `detail`
            # column so milknado_run_inline_poll surfaces `worktree_preserved`,
            # matching the sync dispatch path and milknado_run_cancel.
            graph.finish_run(
                run_id,
                status=terminal,
                exit_code=exit_code,
                timed_out=timed_out,
                ended_at=_now_iso(),
                rebased=merge.rebased if merge is not None else None,
                detail=merge.worktree_preserved if merge is not None else None,
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
        # run dir never carries a stale cancel request into the next run. The
        # tmux path's staged brief and rc file are equally transient — the log
        # (and a failed run's preserved window) carry the diagnostics.
        _clear_cancel(rdir, run_id)
        for staged in (_brief_path(rdir, run_id), _exit_code_path(rdir, run_id)):
            with suppress(FileNotFoundError):
                staged.unlink()


def start_headless_async(
    project_root: Path,
    node_id: int,
    brief: str,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    run_id: str | None = None,
    *,
    default_cmd: str,
    tmux: TmuxAdapter | None = None,
    cwd: Path | None = None,
    merge_ctx: IsolateContext | None = None,
) -> AsyncStartRef:
    argv = _resolve_worker_cmd(worker_cmd, default_cmd)
    # The caller (milknado_run_inline_start) claims the node under a run_id before
    # spawning, then hands that same id here so the node row and the run row agree
    # on the fence; standalone callers let us mint one. Under ISOLATE the caller
    # also created the worktree and passes its path as `cwd` plus a `merge_ctx` for
    # the deferred (post-exit) merge-back the worker thread runs.
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
        args=(project_root, log_path, brief, argv, timeout_seconds, base_state, tmux),
        kwargs={"cwd": cwd, "merge_ctx": merge_ctx},
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
