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
    args = parser.parse_args(argv)

    from milknado._mcp_core import open_graph
    from milknado.adapters import CrgAdapter, GitAdapter, LoopAdapter
    from milknado.domains.common.flavor_profile import resolve_flavor_profile
    from milknado.domains.execution import (
        ExecutionConfig,
        Executor,
        run_node_to_completion,
    )
    from milknado.domains.execution.run_loop._logging import configure_run_logging

    root = Path(args.project_root)
    from milknado.adapters.loop import NO_GATES_CONFIGURED_MESSAGE

    with configure_run_logging(root):
        graph, cfg = open_graph(root)
        node = graph.get_node(args.node_id)
        profile = resolve_flavor_profile(cfg, node.flavor if node is not None else None)

        if profile.quality_gates is None:
            _logger.error(
                "preflight failed for node %s: %s", args.node_id, NO_GATES_CONFIGURED_MESSAGE
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
            graph.close()
            return 1

        try:
            try:
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
                feature_branch = git.current_branch()
                outcome = run_node_to_completion(
                    executor,
                    ralph,
                    args.node_id,
                    exec_config,
                    feature_branch,
                    args.timeout,
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
                return 0 if outcome.success else 1
            except Exception as exc:
                _logger.exception("ralph node runner failed for node %s", args.node_id)
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
