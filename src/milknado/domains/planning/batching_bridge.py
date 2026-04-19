from __future__ import annotations

from typing import TYPE_CHECKING

from milknado.domains.batching import plan_batches

if TYPE_CHECKING:
    from pathlib import Path

    from milknado.domains.batching import Batch, BatchPlan
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.graph import MikadoGraph
    from milknado.domains.planning.manifest import PlanChangeManifest

_DESCRIPTION_FILE_LIMIT = 3


def run_batching(
    manifest: PlanChangeManifest,
    crg: CrgPort | None,
    root: Path,
    *,
    budget: int = 70_000,
    time_limit_s: float = 10.0,
) -> BatchPlan:
    return plan_batches(
        manifest.changes,
        budget=budget,
        crg=crg,
        new_relationships=manifest.new_relationships,
        time_limit_s=time_limit_s,
        root=root,
    )


def apply_batches_to_graph(
    graph: MikadoGraph,
    plan: BatchPlan,
    manifest: PlanChangeManifest,
    *,
    parent_id: int | None = None,
) -> list[int]:
    graph.record_batch_plan(plan)
    if not plan.batches:
        return []

    paths_by_change = {c.id: c.path for c in manifest.changes}
    input_order = {c.id: i for i, c in enumerate(manifest.changes)}
    node_id_by_batch: dict[int, int] = {}
    created: list[int] = []

    for batch in plan.batches:
        files = _batch_files(batch, paths_by_change, input_order)
        node = graph.add_node(
            _batch_description(batch, files),
            parent_id=None,
            oversized=batch.oversized,
            batch_index=batch.index,
        )
        if batch.depends_on:
            for dep_index in batch.depends_on:
                graph.add_edge(node.id, node_id_by_batch[dep_index])
        elif parent_id is not None:
            graph.add_edge(parent_id, node.id)
        if files:
            graph.set_file_ownership(node.id, files)
        node_id_by_batch[batch.index] = node.id
        created.append(node.id)

    return created


def _batch_files(
    batch: Batch,
    paths_by_change: dict[str, str],
    input_order: dict[str, int],
) -> list[str]:
    seen: set[str] = set()
    ordered = sorted(batch.change_ids, key=lambda cid: input_order.get(cid, 0))
    files: list[str] = []
    for cid in ordered:
        path = paths_by_change.get(cid)
        if path is None or path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def _batch_description(batch: Batch, files: list[str]) -> str:
    prefix = f"Batch {batch.index}"
    if not files:
        return prefix
    shown = files[:_DESCRIPTION_FILE_LIMIT]
    extra = len(files) - len(shown)
    body = ", ".join(shown)
    if extra > 0:
        body = f"{body} (+{extra} more)"
    return f"{prefix}: {body}"
