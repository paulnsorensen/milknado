"""Application-layer policy for MCP batch planning.

The ``milknado_plan_batches`` / ``milknado_plan_apply`` MCP tools stay thin and
call the functions here, which decode the request manifest, wire the CRG adapter,
solve the batch plan, and serialize the result. The entry module constructs no
adapter and holds no batching policy inline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.batching import BatchPlan

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.planning.manifest import PlanChangeManifest

_logger = logging.getLogger(__name__)


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


def plan_batches(
    changes: list[dict],
    budget: int,
    project_root: Path,
    new_relationships: list[dict] | None = None,
    *,
    force_single_batch: bool = False,
) -> dict:
    """Decode, solve, and serialize a batch plan from an MCP planning request."""
    manifest = _decode_mcp_changes(changes, new_relationships)
    plan = _solve_manifest(
        manifest,
        budget,
        project_root,
        force_single_batch=force_single_batch,
    )
    return _plan_to_dict(plan)
