"""Detached headless single-node ralph runner.

Spawned as its own process by `milknado_ralph_run_start` so a worktree-isolated
ralph loop survives the MCP server restarting (hot-reload) or a cloud env being
reclaimed: the loop owns its own process, and node status + worktree path live
in SQLite. It reads the `running` state file the MCP tool wrote, runs the node
to completion, and overwrites the file with the terminal state the poll reads.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from milknado.domains.dispatch._runstate import now_iso, read_state, runs_dir, write_state

_logger = logging.getLogger("milknado")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="milknado._ralph_node_runner")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)

    state_path = Path(args.state_path)
    try:
        base = read_state(state_path)
    except (OSError, ValueError):
        # Rebuild the same fields the MCP tool wrote so terminal writes below
        # keep a stable schema (poll/consumers expect log_path/timeout_seconds).
        base = {
            "run_id": args.run_id,
            "node_id": args.node_id,
            "log_path": str(runs_dir(Path(args.project_root)) / f"{args.run_id}.log"),
            "timeout_seconds": args.timeout,
        }

    try:
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
        finally:
            graph.close()

        write_state(
            state_path,
            {
                **base,
                "status": "done" if outcome.success else "failed",
                "rebased": outcome.success,
                "detail": outcome.detail,
                "ended_at": now_iso(),
            },
        )
        return 0 if outcome.success else 1
    except Exception as exc:
        _logger.exception("ralph node runner failed for node %s", args.node_id)
        msg = f"{type(exc).__name__}: {exc}"
        write_state(
            state_path,
            {
                **base,
                "status": "failed",
                "rebased": False,
                "ended_at": now_iso(),
                # `detail` is the documented poll contract field.
                "detail": msg,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
