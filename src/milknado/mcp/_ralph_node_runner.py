"""Detached headless single-node ralph runner.

Spawned as its own process by `milknado_run_loop_start` so a worktree-isolated
ralph loop survives the MCP server restarting (hot-reload). Node status, worktree
path, and run state all persist in SQLite so a retried run can reconcile state
from an earlier process. The MCP tool inserted the `running` run row before
spawning; this process runs the node to completion and writes the terminal run
row the poll reads.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Protocol, cast

from milknado.domains.common import RunResult
from milknado.domains.dispatch import now_iso, runs_dir

_logger = logging.getLogger("milknado")


class _RunFinisher(Protocol):
    def finish_run(self, run_id: str, result: RunResult) -> bool: ...


class _RunnerArgs(Protocol):
    node_id: int
    project_root: str
    run_id: str
    timeout: float
    target_branch: str
    base_oid: str


def _finish_run(graph: object, root: Path, run_id: str, result: RunResult) -> bool:
    try:
        finish_run = cast(_RunFinisher, graph).finish_run
        written = finish_run(run_id, result)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
    else:
        if written is not False:
            return True
        detail = "finish_run lost its running-row fence"
    _logger.error("ralph terminal persistence failed: run_id=%s detail=%s", run_id, detail)
    try:
        _ = (
            runs_dir(root)
            .joinpath(f"{run_id}.terminal-error")
            .write_text(
                detail + "\n",
                encoding="utf-8",
            )
        )
    except OSError:
        _logger.exception("ralph terminal error sidecar write failed: run_id=%s", run_id)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="milknado.mcp._ralph_node_runner")
    _ = parser.add_argument("--node-id", type=int, required=True)
    _ = parser.add_argument("--project-root", required=True)
    _ = parser.add_argument("--run-id", required=True)
    _ = parser.add_argument("--timeout", type=float, default=1800.0)
    _ = parser.add_argument("--target-branch", required=True)
    _ = parser.add_argument("--base-oid", required=True)
    args = cast(_RunnerArgs, cast(object, parser.parse_args(argv)))

    from milknado.adapters import CrgAdapter, GitAdapter, LoopAdapter
    from milknado.app.project import open_graph
    from milknado.domains.common import resolve_flavor_profile
    from milknado.domains.execution import (
        ExecutionConfig,
        Executor,
        run_node_to_completion,
    )

    root = Path(args.project_root)
    from milknado.domains.execution import NO_GATES_CONFIGURED_MESSAGE

    _logger.info(
        "ralph runner started: run_id=%s node_id=%d target_branch=%s base_oid=%s",
        args.run_id,
        args.node_id,
        args.target_branch,
        args.base_oid,
    )
    graph, cfg = open_graph(root)
    pid = os.getpid()
    graph.set_run_pid(args.run_id, pid)
    graph.set_pid(args.node_id, args.run_id, pid)
    try:
        node = graph.get_node(args.node_id)
        profile = resolve_flavor_profile(cfg, node.flavor if node is not None else None)
        if profile.quality_gates is None:
            _logger.error(
                "ralph preflight failed: run_id=%s node_id=%d error=%s",
                args.run_id,
                args.node_id,
                NO_GATES_CONFIGURED_MESSAGE,
            )
            _ = _finish_run(
                graph,
                root,
                args.run_id,
                RunResult(
                    status="failed",
                    exit_code=1,
                    timed_out=False,
                    ended_at=now_iso(),
                    rebased=False,
                    detail=NO_GATES_CONFIGURED_MESSAGE,
                ),
            )
            return 1
        git = GitAdapter(root)
        ralph = LoopAdapter()
        executor = Executor(graph=graph, git=git, ralph=ralph, crg=CrgAdapter(root))
        exec_config = ExecutionConfig(
            execution_agent=profile.execution_agent,
            quality_gates=profile.quality_gates,
            worktree_pattern=cfg.worktree_pattern,
            project_root=root,
            brief_prepend=profile.brief_prepend,
            commit_footer=cfg.commit_footer,
            review=profile.review,
            review_agent=profile.review_agent,
            review_max_rounds=profile.review_max_rounds,
            on_reject=profile.on_reject,
            session_mode=profile.session_mode,
            completion_timeout_seconds=int(args.timeout),
        )
        outcome = run_node_to_completion(
            executor,
            ralph,
            args.node_id,
            exec_config,
            args.target_branch,
            args.timeout,
            base_oid=args.base_oid,
            parent_run_id=args.run_id,
        )
        terminal_written = _finish_run(
            graph,
            root,
            args.run_id,
            RunResult(
                status="done" if outcome.success else "failed",
                exit_code=0 if outcome.success else 1,
                timed_out=False,
                ended_at=now_iso(),
                rebased=outcome.success,
                detail=outcome.detail,
            ),
        )
        if not terminal_written:
            return 2
        _logger.info(
            "ralph runner terminal: run_id=%s node_id=%d success=%s detail=%s",
            args.run_id,
            args.node_id,
            outcome.success,
            outcome.detail,
        )
        return 0 if outcome.success else 1
    except Exception as exc:
        _logger.exception(
            "ralph runner failed: run_id=%s node_id=%d",
            args.run_id,
            args.node_id,
        )
        _ = _finish_run(
            graph,
            root,
            args.run_id,
            RunResult(
                status="failed",
                exit_code=1,
                timed_out=False,
                ended_at=now_iso(),
                rebased=False,
                detail=f"{type(exc).__name__}: {exc}",
            ),
        )
        return 1
    finally:
        graph.close()


if __name__ == "__main__":
    raise SystemExit(main())
