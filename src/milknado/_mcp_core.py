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
TodoStatus = Literal["pending", "in_progress", "blocked", "done"]


class RunDict(TypedDict):
    """Unified superset schema returned by all five run tools.

    Fields present but nullable when not applicable to the run type:
    - run_id: None only for callers that don't produce a state file (legacy)
    - exit_code / timed_out: None while running or for start tools
    - rebased: None for headless (non-ralph) runs
    - state_path: None for sync runs that predate state-file tracking
    - summary: None for start tools (no log tail until polling)
    """

    run_id: str | None
    node_id: int | None
    status: str | None
    exit_code: int | None
    timed_out: bool | None
    rebased: bool | None
    log_path: str | None
    state_path: str | None
    summary: str | None


mcp = FastMCP(
    "Milknado",
    instructions=(
        "Mikado graph tools: list nodes and add prerequisite nodes. "
        "Set MILKNADO_PROJECT_ROOT or pass project_root to target a repo.\n\n"
        "Ephemeral cloud environments: if you are running in a container that "
        "may be reclaimed on timeout (e.g. Claude Code on the web), persist "
        "in-progress work to your branch before going idle. Force-commit "
        "normally-gitignored state with `git add -f` — including "
        "`.milknado/milknado.db` and its `-wal`/`-shm` sidecars when present "
        "(commit the sidecars too so a detached run reclaimed mid-write, before "
        "its WAL is checkpointed into the main file, isn't lost). The db "
        "auto-migrates on load and stores only relative paths, so it spins up "
        "cleanly in a fresh container — and "
        "push to the working branch immediately. Before opening the final PR, "
        "strip those force-added ignored files back out in a cleanup commit "
        "(or interactive rebase): the merged diff must not contain "
        "`.milknado/`, `*.db`, `ralphs/`, `.claude/`, `.context/`, `.venv/`, "
        "or anything else listed in `.gitignore`."
    ),
)


def resolve_project_root(explicit: str | None) -> Path:
    if explicit and explicit.strip():
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("MILKNADO_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


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
