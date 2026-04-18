"""Milknado MCP stdio server — graph tools for Cursor, Codex, Gemini CLI, etc."""

from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

from milknado.domains.batching import BatchPlan, FileChange, NewRelationship, SymbolRef

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


def _dict_to_file_change(d: dict) -> FileChange:
    symbols = tuple(
        SymbolRef(name=s["name"], file=s["file"])
        for s in d.get("symbols", [])
    )
    return FileChange(
        id=d["id"],
        path=d["path"],
        edit_kind=d.get("edit_kind", "modify"),
        symbols=symbols,
        depends_on=tuple(d.get("depends_on", [])),
    )


_VALID_REASONS = frozenset({"new_file", "new_import", "new_call", "new_type_use"})


def _dict_to_new_relationship(d: dict) -> NewRelationship:
    reason = d["reason"]
    if reason not in _VALID_REASONS:
        raise ValueError(
            f"invalid reason: {reason!r}; expected one of {sorted(_VALID_REASONS)}"
        )
    return NewRelationship(
        source_change_id=d["source_change_id"],
        dependant_change_id=d["dependant_change_id"],
        reason=reason,
    )


def _plan_to_dict(plan: BatchPlan) -> dict:
    return {
        "batches": [
            {
                "index": b.index,
                "change_ids": list(b.change_ids),
                "depends_on": list(b.depends_on),
                "oversized": b.oversized,
            }
            for b in plan.batches
        ],
        "spread_report": [
            {"symbol": {"name": ss.symbol.name, "file": ss.symbol.file}, "spread": ss.spread}
            for ss in plan.spread_report
        ],
        "solver_status": plan.solver_status,
    }


def _plan_batches_impl(
    changes: list[dict],
    budget: int,
    project_root: Path,
    new_relationships: list[dict] | None = None,
) -> dict:
    from milknado.adapters.crg import CrgAdapter
    from milknado.domains.batching import plan_batches
    file_changes = [_dict_to_file_change(c) for c in changes]
    rels = tuple(_dict_to_new_relationship(r) for r in (new_relationships or []))
    plan = plan_batches(
        file_changes, budget, crg=CrgAdapter(project_root), new_relationships=rels, root=project_root
    )
    return _plan_to_dict(plan)


@mcp.tool()
def milknado_plan_batches(
    changes: list[dict],
    budget: int = 70_000,
    project_root: str = "",
    new_relationships: list[dict] | None = None,
) -> dict:
    """Compute token-budgeted, precedence-respecting batches for changes."""
    root = _project_root(project_root or None)
    return _plan_batches_impl(changes, budget, root, new_relationships)


def main() -> None:
    mcp.run()
