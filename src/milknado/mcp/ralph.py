"""Milknado MCP ralph-loop tools — thin registration over milknado.app.ralph.

COORDINATOR-ONLY: never add these to WORKER_ALLOWED_TOOLS — a worker must not be
able to spawn sub-ralph-loops. Unlike milknado_run_inline* (a single-shot worker
piped a brief on stdin, isolated in its own worktree by default and merged back
on success — or run in the shared checkout via worktree=THIS_BRANCH), these
dispatch a node into its own git worktree + branch, iterate the ralph loop until
the node's quality gates pass, then rebase-merge the branch back. The loop runs
in its own detached process so it survives the MCP server restarting (hot-reload)
or a cloud env being reclaimed; run state and node status both live in SQLite (the
`runs` table and the `nodes` table), with log files on the filesystem.

The claim/spawn policy and adapter wiring live in `milknado.app.ralph`; this
module only parses I/O and registers the tools. The claim/spawn helpers are
re-exported here for the tests that exercise them directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from milknado.adapters import TmuxAdapter  # noqa: F401
from milknado.app.ralph import (
    _DEFAULT_RUNNER,
    RalphClaim,
    RalphStartRequest,
    _record_spawn_failure,
    _remove_reclaimed_worktree,
    _resolve_runner_cmd,
    start_ralph_run,
)
from milknado.domains.dispatch import (
    RUN_ID_RE,
    runs_dir,
    tail,
    tail_latest_iteration_log,
)
from milknado.mcp._core import (
    build_run_dict,
    mcp,
    open_graph,
    resolve_project_root,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "_DEFAULT_RUNNER",
    "RalphClaim",
    "RalphStartRequest",
    "_record_spawn_failure",
    "_remove_reclaimed_worktree",
    "_resolve_runner_cmd",
    "milknado_run_loop_poll",
    "milknado_run_loop_start",
]


@mcp.tool()
def milknado_run_loop_start(
    node_id: int,
    runner_cmd: str | None = None,
    timeout_seconds: int = 1800,
    use_tmux: bool = False,
    project_root: str = "",
) -> dict:
    """Start a task node in a detached worktree-backed Ralph loop.

    Returns immediately with a run ID for polling. Refuses concurrent dispatch.
    """
    root = resolve_project_root(project_root or None)
    request = RalphStartRequest(
        node_id=node_id,
        runner_cmd=runner_cmd,
        timeout_seconds=timeout_seconds,
        use_tmux=use_tmux,
        root=root,
    )
    graph, _cfg = open_graph(root)
    try:
        return build_run_dict(start_ralph_run(graph, request))
    finally:
        graph.close()


@mcp.tool()
def milknado_run_loop_poll(run_id: str, project_root: str = "") -> dict:
    """Poll a ralph run from durable state and its persisted logs."""
    root = resolve_project_root(project_root or None)
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    rdir = runs_dir(root)
    graph, _cfg = open_graph(root)
    try:
        state = graph.get_run(run_id)
    finally:
        graph.close()
    if state is None:
        raise ValueError(f"run {run_id!r} not found")
    summary = tail(rdir / f"{run_id}.log")
    if not summary:
        log_path = state.get("log_path")
        if log_path:
            candidate = Path(log_path).resolve()
            if candidate.is_dir() and candidate.is_relative_to(root.resolve()):
                summary = tail_latest_iteration_log(candidate)
    state["summary"] = summary
    return build_run_dict(state)
