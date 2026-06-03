"""Detached headless single-node ralph runner.

Spawned as its own process by `milknado_ralph_run_start` so a worktree-isolated
ralph loop survives the MCP server restarting (hot-reload). Node status, worktree
path, and run state all persist in SQLite so a retried run can reconcile state
from an earlier process. The MCP tool inserted the `running` run row before
spawning; this process runs the node to completion and writes the terminal run
row the poll reads.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
    from milknado.adapters import CrgAdapter, GitAdapter, RalphifyAdapter
    from milknado.domains.execution import (
        ExecutionConfig,
        Executor,
        run_node_to_completion,
    )

    root = Path(args.project_root)
    graph, cfg = open_graph(root)
    try:
        try:
            git = GitAdapter(root)
            ralph = RalphifyAdapter()
            crg = CrgAdapter(root)
            executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)
            exec_config = ExecutionConfig(
                execution_agent=cfg.execution_agent,
                quality_gates=cfg.quality_gates,
                worktree_pattern=cfg.worktree_pattern,
                project_root=root,
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
            events = ralph.poll_progress_events()
            graph.finish_run(
                args.run_id,
                status="done" if outcome.success else "failed",
                exit_code=0 if outcome.success else 1,
                timed_out=False,
                ended_at=now_iso(),
                rebased=outcome.success,
                detail=outcome.detail,
            )
            if events:
                # The run_messages table carries progress; the run row's `detail`
                # stays the human poll string. No consumer reads progress yet
                # (YAGNI) but it is durable for one that does.
                graph.deposit_run_message(
                    args.run_id,
                    "progress",
                    json.dumps(
                        [{"work": e.work, "total": e.total, "message": e.message} for e in events]
                    ),
                    now_iso(),
                )
            return 0 if outcome.success else 1
        except Exception as exc:
            _logger.exception("ralph node runner failed for node %s", args.node_id)
            graph.finish_run(
                args.run_id,
                status="failed",
                exit_code=1,
                timed_out=False,
                ended_at=now_iso(),
                rebased=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
            return 1
    finally:
        graph.close()


if __name__ == "__main__":
    raise SystemExit(main())
