from __future__ import annotations

from typing import TYPE_CHECKING

from milknado.domains.batching import plan_batches

if TYPE_CHECKING:
    from pathlib import Path

    from milknado.domains.batching import Batch, BatchPlan
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.graph import MikadoGraph
    from milknado.domains.planning.manifest import PlanChangeManifest


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
    """Apply a BatchPlan to a MikadoGraph.

    When ``parent_id`` is None (fresh graph), a goal root node is created from
    ``manifest.goal`` / ``manifest.goal_summary`` and root batches attach to it.
    When ``parent_id`` is provided, no goal root is created and root batches
    attach to the given parent instead.

    Empty manifest (no batches): returns ``[goal_root_id]`` when parent_id is
    None, or ``[]`` when parent_id is provided.

    Raises ``ValueError`` if ``parent_id`` is None and ``manifest.goal`` or
    ``manifest.goal_summary`` is empty.
    """
    graph.record_batch_plan(plan)

    attach_to: int | None
    created: list[int] = []

    if parent_id is None:
        if not manifest.goal or not manifest.goal_summary:
            raise ValueError(
                "manifest.goal and manifest.goal_summary must be non-empty when parent_id is None"
            )
        goal_root = graph.add_node(manifest.goal_summary)
        created.append(goal_root.id)
        attach_to = goal_root.id
    else:
        attach_to = parent_id

    if not plan.batches:
        return created

    desc_by_change = {c.id: c.description or c.id for c in manifest.changes}
    paths_by_change = {c.id: c.path for c in manifest.changes}
    input_order = {c.id: i for i, c in enumerate(manifest.changes)}
    node_id_by_batch: dict[int, int] = {}
    depended_on: set[int] = {dep for b in plan.batches for dep in b.depends_on}

    for batch in plan.batches:
        files = _batch_files(batch, paths_by_change, input_order)
        node = graph.add_node(
            _batch_description(batch, desc_by_change),
            parent_id=None,
            oversized=batch.oversized,
            batch_index=batch.index,
        )
        if batch.depends_on:
            sorted_deps = sorted(batch.depends_on)
            primary_parent = node_id_by_batch[sorted_deps[0]]
            for dep_index in batch.depends_on:
                graph.add_edge(node.id, node_id_by_batch[dep_index])
        else:
            primary_parent = attach_to
        graph.set_parent_id(node.id, primary_parent)
        if files:
            graph.set_file_ownership(node.id, files)
        node_id_by_batch[batch.index] = node.id
        created.append(node.id)

    for batch in plan.batches:
        if batch.index not in depended_on:
            graph.add_edge(attach_to, node_id_by_batch[batch.index])

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


def _batch_description(
    batch: Batch,
    desc_by_change: dict[str, str],
) -> str:
    lines = []
    for i, cid in enumerate(batch.change_ids, start=1):
        text = desc_by_change.get(cid) or cid
        lines.append(f"{i}. {text}")
    return "\n".join(lines) if lines else f"Batch {batch.index}"
