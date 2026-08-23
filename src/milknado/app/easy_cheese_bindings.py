from __future__ import annotations

from dataclasses import dataclass

from easy_cheese_schemas import SourceCurdRef

from milknado.app.easy_cheese_execution import AppliedMilknadoPlan
from milknado.domains.common import MikadoNode
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import MilknadoPlanV2, render_batch_description


@dataclass(frozen=True)
class CurdExecution:
    source_curd_ref: SourceCurdRef
    items: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class BoundMilknadoExecution:
    items_by_curd: dict[str, CurdExecution]
    nodes: dict[int, MikadoNode]
    expected_criteria: dict[int, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class _IndexedPlanItems:
    known_change_ids: frozenset[str]
    items_by_curd: dict[str, tuple[tuple[str, SourceCurdRef], ...]]
    curd_by_change: dict[str, str]
    path_by_change: dict[str, str]
    change_order: dict[str, int]


def _index_plan_items(plan: MilknadoPlanV2) -> _IndexedPlanItems:
    grouped: dict[str, list[tuple[str, SourceCurdRef]]] = {
        curd.curd_id: [] for curd in plan.semantic_plan.curds
    }
    curd_by_change: dict[str, str] = {}
    path_by_change: dict[str, str] = {}
    change_order: dict[str, int] = {}
    for order, item in enumerate(plan.items):
        curd_id = item.source_curd_ref.curd_id
        if curd_id not in grouped:
            raise ValueError(f"plan item refers to unknown source curd {curd_id!r}")
        change_id = item.change.id
        if change_id in curd_by_change:
            raise ValueError(f"plan contains duplicate change ID {change_id!r}")
        grouped[curd_id].append((change_id, item.source_curd_ref))
        curd_by_change[change_id] = curd_id
        path_by_change[change_id] = item.change.path
        change_order[change_id] = order
    return _IndexedPlanItems(
        frozenset(curd_by_change),
        {curd_id: tuple(items) for curd_id, items in grouped.items()},
        curd_by_change,
        path_by_change,
        change_order,
    )


def _expected_criteria_by_node(
    plan: MilknadoPlanV2,
    indexed: _IndexedPlanItems,
    application: AppliedMilknadoPlan,
) -> dict[int, tuple[tuple[str, str], ...]]:
    criteria_by_curd = {
        curd.curd_id: tuple(criterion.criterion_id for criterion in curd.criteria)
        for curd in plan.semantic_plan.curds
    }
    expected: dict[int, tuple[tuple[str, str], ...]] = {}
    for batch, node_id in zip(
        application.batch_plan.batches, application.task_node_ids, strict=True
    ):
        curd_ids = {indexed.curd_by_change[change_id] for change_id in batch.change_ids}
        expected[node_id] = tuple(
            (curd.curd_id, criterion_id)
            for curd in plan.semantic_plan.curds
            if curd.curd_id in curd_ids
            for criterion_id in criteria_by_curd[curd.curd_id]
        )
    return expected


def _validate_application(
    graph: MikadoGraph,
    plan: MilknadoPlanV2,
    indexed: _IndexedPlanItems,
    application: AppliedMilknadoPlan,
) -> tuple[dict[str, int], dict[int, MikadoNode]]:
    if not isinstance(application, AppliedMilknadoPlan):
        raise TypeError("application must be an AppliedMilknadoPlan")
    if application.source_plan_ref != plan.source_plan_ref:
        raise ValueError("applied plan source reference does not match imported plan")

    batch_change_ids = tuple(
        change_id for batch in application.batch_plan.batches for change_id in batch.change_ids
    )
    if len(set(batch_change_ids)) != len(batch_change_ids):
        raise ValueError("applied batch plan contains duplicate change IDs")
    if set(batch_change_ids) != indexed.known_change_ids:
        missing = sorted(indexed.known_change_ids - set(batch_change_ids))
        unknown = sorted(set(batch_change_ids) - indexed.known_change_ids)
        raise ValueError(
            f"applied batch coverage mismatch: missing={missing!r}, unknown={unknown!r}"
        )

    requested_ids = application.task_node_ids
    nodes = {node.id: node for node in graph.get_nodes(requested_ids)}
    missing_nodes = [node_id for node_id in requested_ids if node_id not in nodes]
    if missing_nodes:
        raise ValueError(f"applied plan refers to nonexistent Mikado node {missing_nodes[0]}")
    ownership = graph.get_file_ownership_map(requested_ids)

    bound_changes: dict[str, int] = {}
    for batch, node_id in zip(
        application.batch_plan.batches, application.task_node_ids, strict=True
    ):
        node = nodes[node_id]
        if node.batch_index != batch.index:
            raise ValueError(
                f"node {node_id} batch index {node.batch_index!r} does not match {batch.index}"
            )
        expected_description = render_batch_description(batch, plan.manifest)
        if node.description != expected_description:
            raise ValueError(f"node {node_id} description does not match its applied batch")
        ordered_changes = sorted(batch.change_ids, key=indexed.change_order.__getitem__)
        expected_paths = tuple(
            dict.fromkeys(indexed.path_by_change[change_id] for change_id in ordered_changes)
        )
        actual_paths = tuple(ownership.get(node_id, ()))
        if actual_paths != expected_paths:
            raise ValueError(
                f"node {node_id} file ownership does not match its applied changes: "
                f"expected={expected_paths!r}, actual={actual_paths!r}"
            )
        for change_id in batch.change_ids:
            bound_changes[change_id] = node_id
    return bound_changes, nodes


def bind_milknado_execution(
    graph: MikadoGraph,
    plan: MilknadoPlanV2,
    application: AppliedMilknadoPlan,
) -> BoundMilknadoExecution:
    indexed = _index_plan_items(plan)
    bound_changes, nodes = _validate_application(graph, plan, indexed, application)
    items_by_curd = {
        curd_id: CurdExecution(
            items[0][1],
            tuple((change_id, bound_changes[change_id]) for change_id, _ref in items),
        )
        for curd_id, items in indexed.items_by_curd.items()
        if items
    }
    return BoundMilknadoExecution(
        items_by_curd,
        nodes,
        _expected_criteria_by_node(plan, indexed, application),
    )
