"""Milknado MCP stdio server — graph tools for Cursor, Codex, Gemini CLI, etc."""

from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP(
    "Milknado",
    instructions=(
        "Mikado graph tools: list nodes and add prerequisite nodes. "
        "Set MILKNADO_PROJECT_ROOT or pass project_root to target a repo."
    ),
)


def _project_root(explicit: str | None) -> Path:
    if explicit and explicit.strip():
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("MILKNADO_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


def _open_graph(root: Path):
    from milknado.domains.common import default_config, load_config
    from milknado.domains.graph import MikadoGraph

    cfg_path = root / "milknado.toml"
    cfg = load_config(cfg_path) if cfg_path.exists() else default_config(root)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    return MikadoGraph(cfg.db_path), cfg


@mcp.tool()
def milknado_graph_summary(project_root: str = "") -> str:
    """Return Mikado nodes (id, status, description) for the given project."""
    root = _project_root(project_root or None)
    graph, _cfg = _open_graph(root)
    try:
        nodes = graph.get_all_nodes()
        if not nodes:
            return "(empty graph)"
        lines = [
            f"id={n.id} status={n.status.value} desc={n.description[:120]!r}"
            for n in nodes
        ]
        return "\n".join(lines)
    finally:
        graph.close()


@mcp.tool()
def milknado_add_node(
    description: str,
    parent_id: int | None = None,
    project_root: str = "",
) -> str:
    """Add a Mikado node; optional parent_id links a prerequisite edge."""
    root = _project_root(project_root or None)
    graph, _cfg = _open_graph(root)
    try:
        node = graph.add_node(description, parent_id=parent_id)
        return f"created node id={node.id} description={node.description!r}"
    finally:
        graph.close()


def main() -> None:
    mcp.run()
