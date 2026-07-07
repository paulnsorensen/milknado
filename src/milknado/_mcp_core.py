"""Shared FastMCP instance and project-root/graph helpers for the MCP tools.

Lives in its own module so the tool modules (`mcp_server`, `mcp_todo`) can both
register against the same `mcp` instance without importing each other — which
would form an import cycle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, TypedDict

from fastmcp import FastMCP

from milknado.domains.common import NodeKind, NodeStatus

Kind = Literal["roadmap", "goal", "task"]
Flavor = str  # validated at runtime against BUILTIN_FLAVORS ∪ TOML-declared names (ADR-004)
TodoStatus = Literal["pending", "in_progress", "blocked", "done"]


class RunDict(TypedDict):
    """Unified superset schema returned by all five run tools.

    Fields present but nullable when not applicable to the run type:
    - run_id: None only for callers that don't produce a state file (legacy)
    - exit_code: None while running or for start tools
    - timed_out: False while running; True only once failed by timeout; None for start tools
    - rebased: None for headless (non-ralph) runs
    - summary: None for start tools (no log tail until polling)
    """

    run_id: str | None
    node_id: int | None
    status: str | None
    exit_code: int | None
    timed_out: bool | None
    rebased: bool | None
    log_path: str | None
    summary: str | None


mcp = FastMCP(
    "Milknado",
    instructions=(
        "Mikado graph execution tools: graph CRUD, batch planning, worker dispatch, "
        "detached ralph runs, run polling/cancel, and roadmap import/export. "
        "Set MILKNADO_PROJECT_ROOT or pass project_root to target a repo.\n\n"
        "Ephemeral cloud containers may be reclaimed on timeout. Before going idle, "
        "persist in-progress work to your branch; if you need run state, force-add "
        "`.milknado/milknado.db` plus its `-wal`/`-shm` sidecars, then remove "
        "ignored state (`.milknado/`, `*.db`, `ralphs/`, `.claude/`, `.context/`, "
        "`.venv/`, etc.) before opening the final PR."
    ),
)


def _worktree_main_checkout(cwd: Path) -> Path | None:
    """If cwd is a linked git worktree, return the main checkout root; else None."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError):
        return None
    if result.returncode != 0:
        return None
    common_dir = result.stdout.strip()
    # An absolute path means we're in a linked worktree: the common dir lives
    # inside the main checkout's .git. The main checkout root is its parent
    # (common_dir ends in .git, so parent is the main checkout).
    if not os.path.isabs(common_dir):
        return None  # relative == main checkout (e.g. ".git") — no hop needed
    return Path(common_dir).parent.resolve()


def resolve_project_root(explicit: str | None) -> Path:
    if explicit and explicit.strip():
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("MILKNADO_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    main = _worktree_main_checkout(cwd)
    return main if main is not None else cwd


def _check_ancestor_goal_not_claimed(graph, node_id: int) -> None:  # noqa: ANN001
    """Raise ValueError if an ancestor goal is claimed by a different live coordinator.

    Same-process dispatch (same pid) is exempt: the current process is the
    coordinator that holds the claim, so sibling tasks may proceed.
    """
    claim = graph.ancestor_goal_claimed_by_other(node_id)
    if claim is None:
        return
    # Same coordinator process: the current pid owns the claim → siblings allowed.
    if claim["pid"] == os.getpid():
        return
    raise ValueError(
        f"node {node_id}'s ancestor goal is claimed by a different run "
        f"({claim['run_id']!r}); dispatch refused"
    )


def _claim_ancestor_goal_for_dispatch(graph, node_id: int, run_id: str) -> None:  # noqa: ANN001
    """Claim the closest ancestor goal for run_id+pid before dispatch.

    No-op when the node has no ancestor goal.  Called after
    _check_ancestor_goal_not_claimed so a live foreign claimant is already
    refused before this write.
    """
    from milknado.domains.dispatch import now_iso

    graph.claim_ancestor_goal(node_id, run_id, os.getpid(), now=now_iso())


def open_graph(root: Path):
    from milknado.domains.common import default_config, load_config
    from milknado.domains.graph import MikadoGraph
    from milknado.plugins import discover_entry_point_plugins, load_plugins

    cfg_path = root / "milknado.toml"
    cfg = load_config(cfg_path) if cfg_path.exists() else default_config(root)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    plugins = [*load_plugins(cfg.plugins), *discover_entry_point_plugins()]
    return MikadoGraph(cfg.db_path, plugins=plugins), cfg


_TODO_STATUS_MAP = {
    "pending": NodeStatus.PENDING,
    "in_progress": NodeStatus.RUNNING,
    "blocked": NodeStatus.BLOCKED,
    "done": NodeStatus.DONE,
}


def _parse_kind(value: str) -> NodeKind:
    try:
        return NodeKind(value)
    except ValueError as exc:
        valid = sorted(k.value for k in NodeKind)
        raise ValueError(f"invalid kind {value!r}; expected one of {valid}") from exc


def _parse_todo_status(value: str) -> NodeStatus:
    if value not in _TODO_STATUS_MAP:
        raise ValueError(f"invalid status {value!r}; expected one of {sorted(_TODO_STATUS_MAP)}")
    return _TODO_STATUS_MAP[value]


def _parse_flavor(value: str, registry: frozenset[str]) -> str:
    if value not in registry:
        raise ValueError(f"invalid flavor {value!r}; expected one of {sorted(registry)}")
    return value
