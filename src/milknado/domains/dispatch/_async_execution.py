"""Execution and proof helpers for detached dispatch."""

from __future__ import annotations

import secrets
from pathlib import Path

from milknado.domains.dispatch._async_types import AsyncWorkerContext
from milknado.domains.dispatch._runstate import brief_path as _brief_path
from milknado.domains.dispatch._runstate import exec_receipt_path
from milknado.domains.dispatch._runstate import exit_code_path as _exit_code_path
from milknado.domains.dispatch._runstate import now_iso as _now_iso
from milknado.domains.dispatch._sync_completion import confirm_execution
from milknado.domains.dispatch.ports import RunWindow, TmuxPort
from milknado.domains.dispatch.runner import build_worker_env, execute_cancellable
from milknado.domains.dispatch.tmux_run import execute_in_window as _execute_in_window
from milknado.domains.graph import MikadoGraph


def run_worker_process(
    context: AsyncWorkerContext, graph: MikadoGraph, rdir: Path
) -> tuple[int, bool, bool]:
    request = context.request
    if context.tmux is None:
        return execute_cancellable(
            request.cwd,
            request.node_id,
            context.log_path,
            request.brief,
            list(context.argv),
            request.timeout_seconds,
            runs_dir=rdir,
            run_id=request.run_id,
            project_root=request.project_root,
            process=context.process,
            on_started=lambda pid: confirm_execution(graph, request.run_id, pid),
        )
    return _run_in_tmux_window(context, graph, rdir, context.tmux)


def _run_in_tmux_window(
    context: AsyncWorkerContext, graph: MikadoGraph, rdir: Path, tmux: TmuxPort
) -> tuple[int, bool, bool]:
    request = context.request
    run_id = request.run_id
    staged_brief = _brief_path(rdir, run_id)
    _ = staged_brief.write_text(request.brief, encoding="utf-8")
    context.log_path.touch()
    nonce = secrets.token_hex(16)
    _ = graph.runs.deposit_message(run_id, "tmux_exec_nonce", nonce, _now_iso())
    receipt = exec_receipt_path(graph.db_path, run_id, nonce)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    window = RunWindow(
        run_id=run_id,
        argv=context.argv,
        cwd=request.cwd,
        log_path=context.log_path,
        exit_code_path=_exit_code_path(rdir, run_id),
        env=build_worker_env(
            {
                "MILKNADO_NODE_ID": str(request.node_id),
                "MILKNADO_RUN_ID": run_id,
                "MILKNADO_PROJECT_ROOT": str(request.project_root.resolve()),
            },
            executable=context.argv[0],
        ),
        brief_path=staged_brief,
        exec_receipt_path=receipt,
        exec_nonce=nonce,
    )
    return _execute_in_window(
        tmux,
        window,
        request.timeout_seconds,
        rdir,
        on_start=lambda pane_pid: graph.runs.set_pid(run_id, pane_pid),
    )
