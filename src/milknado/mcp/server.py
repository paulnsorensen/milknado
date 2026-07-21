"""Milknado MCP stdio server — graph tools for Cursor, Codex, Gemini CLI, etc."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.planning.manifest import PlanChangeManifest

from milknado.domains.batching import DUMB_ZONE_BUDGET, BatchPlan
from milknado.mcp._core import (
    Flavor,
    Kind,
    TodoStatus,
    _parse_flavor,
    _parse_kind,
    _parse_todo_status,
    mcp,
    open_graph,
    resolve_project_root,
)

__all__ = ["main", "mcp", "open_graph", "resolve_project_root"]

_logger = logging.getLogger(__name__)


@mcp.tool()
def milknado_graph_summary(
    project_root: str = "",
    status: TodoStatus | None = None,
    kind: Kind | None = None,
    flavor: Flavor | None = None,
    page: tuple[int, int] = (100, 0),
) -> dict:
    """Return a bounded page of Mikado nodes filtered in the graph query.

    ``page`` is ``(limit, offset)``. Each node includes id, status, and
    description; no matches returns an empty nodes list.
    """
    want_status = _parse_todo_status(status) if status is not None else None
    want_kind = _parse_kind(kind) if kind is not None else None
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        want_flavor = _parse_flavor(flavor, cfg.flavor_registry) if flavor is not None else None
        return {
            "nodes": graph.get_node_summaries(
                status=want_status,
                kind=want_kind,
                flavor=want_flavor,
                page=page,
            )
        }
    finally:
        graph.close()


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


def _try_crg(project_root: Path) -> CrgPort | None:
    # #71: log CRG failures instead of swallowing them silently
    from milknado.adapters.crg import CrgAdapter

    try:
        crg = CrgAdapter(project_root)
        crg.ensure_graph(project_root)
    except Exception as exc:
        _logger.warning("CRG unavailable for MCP planning, proceeding without graph: %s", exc)
        return None
    return crg


def _solve_manifest(
    manifest: PlanChangeManifest,
    budget: int,
    project_root: Path,
    *,
    force_single_batch: bool,
) -> BatchPlan:
    from milknado.domains.batching import MEGA_BATCH_THRESHOLD, plan_batches
    from milknado.domains.common import MegaBatchAborted
    from milknado.domains.planning import record_batch_snapshot

    plan = plan_batches(
        manifest.changes,
        budget,
        crg=_try_crg(project_root),
        new_relationships=manifest.new_relationships,
        root=project_root,
    )
    count = plan.mega_batch_change_count
    if count is not None and not force_single_batch:
        raise MegaBatchAborted(change_count=count, threshold=MEGA_BATCH_THRESHOLD)
    record_batch_snapshot(project_root, manifest, plan)
    return plan


def _decode_mcp_changes(
    changes: list[dict],
    new_relationships: list[dict] | None,
) -> PlanChangeManifest:
    from milknado.domains.planning import MANIFEST_VERSION, decode_manifest

    return decode_manifest(
        {
            "manifest_version": MANIFEST_VERSION,
            "goal": "(mcp)",
            "goal_summary": "MCP planning request",
            "spec_path": None,
            "changes": changes,
            "new_relationships": new_relationships or [],
        }
    )


def _plan_batches_impl(
    changes: list[dict],
    budget: int,
    project_root: Path,
    new_relationships: list[dict] | None = None,
    *,
    force_single_batch: bool = False,
) -> dict:
    manifest = _decode_mcp_changes(changes, new_relationships)
    plan = _solve_manifest(
        manifest,
        budget,
        project_root,
        force_single_batch=force_single_batch,
    )
    return _plan_to_dict(plan)


@mcp.tool()
def milknado_plan_batches(
    changes: list[dict],
    budget: int = DUMB_ZONE_BUDGET,
    project_root: str = "",
    new_relationships: list[dict] | None = None,
    force_single_batch: bool = False,
) -> dict:
    """Compute token-budgeted, precedence-respecting batches for changes."""
    root = resolve_project_root(project_root or None)
    return _plan_batches_impl(
        changes, budget, root, new_relationships, force_single_batch=force_single_batch
    )


@mcp.tool()
def milknado_plan_apply(
    manifest: dict,
    project_root: str = "",
    parent_id: int | None = None,
    budget: int = DUMB_ZONE_BUDGET,
    force_single_batch: bool = False,
) -> dict:
    """Apply a caller-produced milknado.plan.v2 manifest to the graph as Mikado nodes.

    Solves the manifest into precedence-respecting batches and writes them as nodes:
    parent_id=None creates a GOAL root from goal_summary with TASK children; a valid
    parent_id attaches TASK children under it. Returns {nodes_created, graph_summary}.
    """
    from milknado.domains.planning import apply_batches_to_graph, decode_manifest

    root = resolve_project_root(project_root or None)
    parsed = decode_manifest(manifest)
    plan = _solve_manifest(parsed, budget, root, force_single_batch=force_single_batch)
    graph, _cfg = open_graph(root)
    try:
        created = apply_batches_to_graph(graph, plan, parsed, parent_id=parent_id)
        summary = {"nodes": graph.get_node_summaries()}
    finally:
        graph.close()
    return {"nodes_created": created, "graph_summary": summary}


def main() -> None:
    from milknado.process_logging import configure_stderr_logging

    # the MCP server speaks stdio — logging must never write to stdout, only stderr/file.
    configure_stderr_logging()

    # Importing each tool module registers its @mcp.tool()s on the shared instance.
    from milknado.mcp import (  # noqa: F401
        github,
        node,
        ralph,
        run,
        todo,
        todo_mutate,
        wiki,
    )

    mcp.run()
