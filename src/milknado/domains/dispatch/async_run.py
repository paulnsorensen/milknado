"""Detached-headless async dispatch: background thread worker, start, and poll."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import GitPort, RunResult
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
from milknado.domains.dispatch.ports import GraphSessionPort, ProcessPort, RunWindow, TmuxPort
from milknado.domains.dispatch.runner import (
    AsyncStartRef,
    _execute_cancellable,
    _resolve_worker_cmd,
    build_worker_env,
)
from milknado.domains.dispatch.tmux_run import execute_in_window as _execute_in_window

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncRunRequest:
    project_root: Path
    node_id: int
    brief: str
    worker_cmd: str | None
    timeout_seconds: int
    run_id: str
    default_cmd: str
    cwd: Path
    merge_ctx: IsolateContext | None = None


@dataclass(frozen=True)
class AsyncWorkerContext:
    request: AsyncRunRequest
    log_path: Path
    argv: tuple[str, ...]
    graph_sessions: GraphSessionPort
    git: GitPort
    process: ProcessPort
    tmux: TmuxPort | None = None


def _run_worker_process(
    cwd: Path,
    log_path: Path,
    brief: str,
    argv: list[str],
    *,
    timeout: int,
    base_state: dict,
    rdir: Path,
    tmux: TmuxPort | None,
    graph,  # noqa: ANN001
    process: ProcessPort,
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
            project_root=Path(base_state["project_root"]),
            process=process,
            on_started=lambda pid: graph.set_run_pid(base_state["run_id"], pid),
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
    tmux: TmuxPort,
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
        env=build_worker_env(
            {
                "MILKNADO_NODE_ID": str(base_state["node_id"]),
                "MILKNADO_RUN_ID": run_id,
                "MILKNADO_PROJECT_ROOT": str(Path(base_state["project_root"]).resolve()),
            }
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
    git: GitPort,
    project_root: Path,
    merge_ctx: IsolateContext | None,
    terminal: str,
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
    result = merge_back_isolated(git, project_root, merge_ctx)
    if result.worktree_preserved is not None:
        _logger.warning(
            "ISOLATE merge-back for branch %s did not tear down; preserved worktree %s",
            merge_ctx.worker_branch,
            result.worktree_preserved,
        )
    return result


def _async_worker(context: AsyncWorkerContext) -> None:
    request = context.request
    project_root = request.project_root
    log_path = context.log_path
    run_id = request.run_id
    rdir = _runs_dir(project_root)
    graph = None
    try:
        graph, _cfg = context.graph_sessions.open_graph(project_root)
        _logger.info("async dispatch started: run_id=%s node_id=%d", run_id, request.node_id)
        exit_code, timed_out, cancelled = _run_worker_process(
            request.cwd,
            log_path,
            request.brief,
            list(context.argv),
            timeout=request.timeout_seconds,
            base_state={
                "run_id": run_id,
                "node_id": request.node_id,
                "project_root": str(project_root),
            },
            rdir=rdir,
            tmux=context.tmux,
            process=context.process,
            graph=graph,
        )
        # The worker owns the terminal write for a cancelled run: run_cancel only
        # requests cancellation, so finalizing here (not there) is what closes the
        # state-clobber race the old signal-then-overwrite path left open.
        if cancelled:
            if (
                graph.finish_run(
                    run_id,
                    RunResult(
                        status="failed",
                        exit_code=-1,
                        timed_out=timed_out,
                        ended_at=_now_iso(),
                        error="cancelled",
                    ),
                )
                is False
            ):
                raise RuntimeError(f"terminal run write lost its fence for {run_id}")
        else:
            terminal = "done" if exit_code == 0 and not timed_out else "failed"
            merge = _async_merge_back(context.git, project_root, request.merge_ctx, terminal)
            if merge is not None and merge.rebased is False:
                # ISOLATE merge_back requested but the branch did not land: the
                # deliverable never reached the dispatch branch, so this is a real
                # failure — do not reconcile the node DONE off a clean worker exit.
                terminal = "failed"
            # Persist a preserved (un-torn-down) worktree via the run row's `detail`
            # column so milknado_run_inline_poll surfaces `worktree_preserved`,
            # matching the sync dispatch path and milknado_run_cancel.
            if (
                graph.finish_run(
                    run_id,
                    RunResult(
                        status=terminal,
                        exit_code=exit_code,
                        timed_out=timed_out,
                        ended_at=_now_iso(),
                        rebased=merge.rebased if merge is not None else None,
                        detail=merge.worktree_preserved if merge is not None else None,
                    ),
                )
                is False
            ):
                raise RuntimeError(f"terminal run write lost its fence for {run_id}")
            _logger.info(
                "async dispatch terminal: run_id=%s node_id=%d status=%s "
                "exit_code=%d timed_out=%s cancelled=%s rebased=%s",
                run_id,
                request.node_id,
                terminal,
                exit_code,
                timed_out,
                cancelled,
                merge.rebased if merge is not None else None,
            )
    except Exception as exc:
        original_detail = f"{type(exc).__name__}: {exc}"
        _logger.warning(
            "async worker for run %s raised %s: %s", run_id, type(exc).__name__, exc, exc_info=True
        )
        # Spawn failures (FileNotFoundError on bad worker_cmd, OSError, etc.) or
        # write failures must not leave the run stuck on "running" — that would
        # silently lock out the node forever. Append to the log so any partial
        # subprocess output that did make it to disk is preserved.
        try:
            with log_path.open("a", encoding="utf-8") as log_fh:
                log_fh.write(f"\n--- async worker raised: {original_detail} ---\n")
        except OSError:
            # Annotating the log is best-effort; the terminal run write below is
            # what actually releases the node, so a log-write failure here is
            # deliberately ignored.
            pass
        if graph is not None:
            terminal_detail = original_detail
            try:
                written = graph.finish_run(
                    run_id,
                    RunResult(
                        status="failed",
                        exit_code=-1,
                        timed_out=False,
                        ended_at=_now_iso(),
                        error=original_detail,
                    ),
                )
            except Exception as persist_exc:
                written = False
                terminal_detail = (
                    f"{original_detail}; terminal persistence raised: "
                    f"{type(persist_exc).__name__}: {persist_exc}"
                )
            if written is False:
                with suppress(OSError):
                    (rdir / f"{run_id}.terminal-error").write_text(
                        f"run_id={run_id}\n"
                        f"node_id={request.node_id}\n"
                        f"log_path={log_path}\n"
                        f"{terminal_detail}\n",
                        encoding="utf-8",
                    )
    finally:
        if graph is not None:
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
    request: AsyncRunRequest,
    graph_sessions: GraphSessionPort,
    git: GitPort,
    process: ProcessPort,
    tmux: TmuxPort | None = None,
) -> AsyncStartRef:
    argv = tuple(_resolve_worker_cmd(request.worker_cmd, request.default_cmd))
    runs = _runs_dir(request.project_root)
    log_path = runs / f"{request.run_id}.log"
    graph, _cfg = graph_sessions.open_graph(request.project_root)
    try:
        graph.start_run(
            request.run_id,
            request.node_id,
            str(log_path),
            _now_iso(),
            request.timeout_seconds,
        )
    finally:
        graph.close()
    context = AsyncWorkerContext(
        request=request,
        log_path=log_path,
        argv=argv,
        graph_sessions=graph_sessions,
        git=git,
        process=process,
        tmux=tmux,
    )
    thread = threading.Thread(
        target=_async_worker,
        args=(context,),
        daemon=True,
        name=f"milknado-async-{request.run_id}",
    )
    thread.start()
    return AsyncStartRef(run_id=request.run_id, log_path=log_path)


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
    terminal_error = _runs_dir(project_root) / f"{run_id}.terminal-error"
    if terminal_error.exists():
        state["error"] = terminal_error.read_text(encoding="utf-8").strip()
        state["terminal_persistence"] = "degraded"
    return state
