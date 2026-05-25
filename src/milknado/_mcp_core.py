"""Shared FastMCP instance and project-root/graph helpers for the MCP tools.

Lives in its own module so the tool modules (`mcp_server`, `mcp_todo`) can both
register against the same `mcp` instance without importing each other — which
would form an import cycle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP

Kind = Literal["roadmap", "goal", "task"]
TodoStatus = Literal["pending", "in_progress", "blocked", "done"]

mcp = FastMCP(
    "Milknado",
    instructions=(
        "Mikado graph tools: list nodes and add prerequisite nodes. "
        "Set MILKNADO_PROJECT_ROOT or pass project_root to target a repo.\n\n"
        "Ephemeral cloud environments: if you are running in a container that "
        "may be reclaimed on timeout (e.g. Claude Code on the web), persist "
        "in-progress work to your branch before going idle. Force-commit "
        "normally-gitignored state with `git add -f` — including "
        "`.milknado/milknado.db`, which auto-migrates on load and stores only "
        "relative paths, so it spins up cleanly in a fresh container — and "
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

    cfg_path = root / "milknado.toml"
    cfg = load_config(cfg_path) if cfg_path.exists() else default_config(root)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    return MikadoGraph(cfg.db_path), cfg
