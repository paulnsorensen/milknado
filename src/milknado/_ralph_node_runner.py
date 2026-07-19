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
from pathlib import Path

from milknado.domains.common import RunResult
from milknado.domains.dispatch import now_iso

_logger = logging.getLogger("milknado")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="milknado._ralph_node_runner")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--target-branch", required=True)
    parser.add_argument("--base-oid", required=True)
    args = parser.parse_args(argv)

    from milknado._mcp_core import open_graph
    from milknado.adapters import CrgAdapter, GitAdapter, LoopAdapter
    from milknado.domains.common.flavor_profile import resolve_flavor_profile
    from milknado.domains.execution import (
        ExecutionConfig,
        Executor,
        run_node_to_completion,
    )

    root = Path(args.project_root)
    from milknado.domains.execution.completion import NO_GATES_CONFIGURED_MESSAGE

    _logger.info(
        "ralph runner started: run_id=%s node_id=%d target_branch=%s base_oid=%s",
        args.run_id,
        args.node_id,
        args.target_branch,
        args.base_oid,
    )
    graph, cfg = open_graph(root)
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
            graph.finish_run(
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
        crg = CrgAdapter(root)
        executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)
        exec_config = ExecutionConfig(
            execution_agent=profile.execution_agent,
            quality_gates=profile.quality_gates,
            worktree_pattern=cfg.worktree_pattern,
            project_root=root,
            commit_footer=cfg.commit_footer,
        )
        outcome = run_node_to_completion(
            executor,
            ralph,
            args.node_id,
            exec_config,
            args.target_branch,
            args.timeout,
            base_oid=args.base_oid,
        )
        graph.finish_run(
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
        graph.finish_run(
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
