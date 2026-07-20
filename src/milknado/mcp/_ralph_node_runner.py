"""Detached headless single-node ralph runner — thin process entry.

Spawned by `milknado_run_loop_start`. Delegates to `milknado.app.ralph.run_node_subprocess`
for all policy; this module only owns argument parsing and the process exit code.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

_logger = logging.getLogger("milknado")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="milknado.mcp._ralph_node_runner")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--target-branch", required=True)
    parser.add_argument("--base-oid", required=True)
    args = parser.parse_args(argv)

    from milknado.app.ralph import RunNodeRequest, run_node_subprocess

    return run_node_subprocess(
        RunNodeRequest(
            root=Path(args.project_root),
            node_id=args.node_id,
            run_id=args.run_id,
            timeout=args.timeout,
            target_branch=args.target_branch,
            base_oid=args.base_oid,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
