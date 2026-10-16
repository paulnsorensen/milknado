"""Detached-headless async dispatch: background thread worker, start, and poll."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from pathlib import Path

from milknado.domains.common import GitPort, RunFenceLostError, RunResult
from milknado.domains.dispatch._async_execution import run_worker_process
from milknado.domains.dispatch._async_types import (
    AsyncRunRequest as AsyncRunRequest,
)
from milknado.domains.dispatch._async_types import (
    AsyncWorkerContext as AsyncWorkerContext,
)
from milknado.domains.dispatch._attempt_completion import (
    AttemptRef,
    settle_cancelled,
    settle_counted_failure,
    settle_preserved,
)
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
from milknado.domains.dispatch._sync_completion import finish_attempt_exception
from milknado.domains.dispatch.isolate import merge_back_if_done
from milknado.domains.dispatch.ports import GraphSessionPort, ProcessPort, TmuxPort
from milknado.domains.dispatch.runner import AsyncStartRef, resolve_worker_cmd
from milknado.domains.graph import FinalizeTaskFailure, MikadoGraph, StartTaskAttempt

_logger = logging.getLogger(__name__)


def _async_worker(context: AsyncWorkerContext) -> None:
    request = context.request
    project_root = request.project_root
    log_path = context.log_path
    run_id = request.run_id
    rdir = _runs_dir(project_root)
    graph: MikadoGraph | None = None
    try:
        graph, _cfg = context.graph_sessions.open_graph(project_root)
        _logger.info("async dispatch started: run_id=%s node_id=%d", run_id, request.node_id)
        exit_code, timed_out, cancelled = run_worker_process(context, graph, rdir)
        if graph.runs.live_workers(run_id=run_id):
            raise RuntimeError(f"worker cleanup unresolved for run {run_id}")
        # The worker owns the terminal write for a cancelled run: run_cancel only
        # requests cancellation, so finalizing here (not there) is what closes the
        # state-clobber race the old signal-then-overwrite path left open.
        ref = AttemptRef(request.node_id, run_id, run_id, run_id)
        if cancelled:
            settle_cancelled(
                graph, ref, RunResult("failed", -1, timed_out, _now_iso(), error="cancelled")
            )
        else:
            terminal = "done" if exit_code == 0 and not timed_out else "failed"
            merge = merge_back_if_done(context.git, project_root, request.merge_ctx, terminal)
            if merge is not None and merge.rebased is False:
                # ISOLATE merge_back requested but the branch did not land: the
                # deliverable never reached the dispatch branch, so this is a real
                # failure — do not reconcile the node DONE off a clean worker exit.
                terminal = "failed"
            # Persist a preserved (un-torn-down) worktree via the run row's `detail`
            # column so milknado_run_inline_poll surfaces `worktree_preserved`,
            # matching the sync dispatch path and milknado_run_cancel.
            result = RunResult(
                status=terminal,
                exit_code=exit_code,
                timed_out=timed_out,
                ended_at=_now_iso(),
                rebased=merge.rebased if merge is not None else None,
                detail=merge.worktree_preserved if merge is not None else None,
            )
            preserved = merge is not None and not merge.rebased and exit_code == 0
            if terminal == "failed" and not preserved:
                failure = FinalizeTaskFailure(request.node_id, run_id, run_id, run_id, result)
                _ = settle_counted_failure(graph, failure)
            else:
                graph.runs.finish(run_id, result)
                if preserved:
                    settle_preserved(graph, ref)
                else:
                    graph.task_attempts.require_nonfailure(
                        ref.attempt_id, ref.owner_run_id, "done"
                    )
            _logger.info(
                (
                    "async dispatch terminal: run_id=%s node_id=%d status=%s "
                    + "exit_code=%d timed_out=%s cancelled=%s rebased=%s"
                ),
                run_id,
                request.node_id,
                terminal,
                exit_code,
                timed_out,
                cancelled,
                merge.rebased if merge is not None else None,
            )
    except RunFenceLostError:
        _logger.info("async worker terminal write lost fence for run %s", run_id)
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
                _ = log_fh.write(f"\n--- async worker raised: {original_detail} ---\n")
        except OSError:
            # Annotating the log is best-effort; the terminal run write below is
            # what actually releases the node, so a log-write failure here is
            # deliberately ignored.
            pass
        if graph is not None:
            try:
                finish_attempt_exception(graph, request, (run_id, context.tmux is None), exc)
            except RunFenceLostError:
                _logger.info("async worker failure run already finalized: run_id=%s", run_id)
            except Exception as persist_exc:
                terminal_detail = (
                    f"{original_detail}; terminal persistence raised: "
                    f"{type(persist_exc).__name__}: {persist_exc}"
                )
                with suppress(OSError):
                    _ = (rdir / f"{run_id}.terminal-error").write_text(
                        f"run_id={run_id}\n"
                        + f"node_id={request.node_id}\n"
                        + f"log_path={log_path}\n"
                        + f"{terminal_detail}\n",
                        encoding="utf-8",
                    )
    finally:
        if request.lease is not None:
            request.lease.release()
        if graph is not None:
            graph.close()
        # Clear the sentinel after terminal write so reused run dirs cannot carry stale cancel.
        # The tmux brief and rc are transient; the log and preserved window carry diagnostics.
        _clear_cancel(rdir, run_id)
        for staged in (_brief_path(rdir, run_id), _exit_code_path(rdir, run_id)):
            with suppress(FileNotFoundError):
                staged.unlink()


def _bind_attempt(graph: MikadoGraph, request: AsyncRunRequest, log_path: Path) -> None:
    """Record the task attempt and its run row before the worker thread starts."""
    _ = graph.task_attempts.start(
        StartTaskAttempt(
            request.node_id,
            request.run_id,
            request.run_id,
            request.run_id,
            request.max_node_attempts,
        )
    )
    graph.runs.start(
        request.run_id,
        request.node_id,
        str(log_path),
        _now_iso(),
        request.timeout_seconds,
        attempt_id=request.run_id,
    )


def start_headless_async(
    request: AsyncRunRequest,
    graph_sessions: GraphSessionPort,
    git: GitPort,
    process: ProcessPort,
    tmux: TmuxPort | None = None,
) -> AsyncStartRef:
    argv = tuple(resolve_worker_cmd(request.worker_cmd, request.default_cmd))
    graph, _cfg = graph_sessions.open_graph(request.project_root)
    try:
        graph.register_controller_master()
        runs = _runs_dir(request.project_root)
        log_path = runs / f"{request.run_id}.log"
        log_path.touch()
        _bind_attempt(graph, request, log_path)
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


def poll_async_run(graph: MikadoGraph, project_root: Path, run_id: str) -> dict[str, object]:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    if (record := graph.runs.get(run_id)) is None:
        raise ValueError(f"run {run_id!r} not found")
    state: dict[str, object] = dict(record)
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
