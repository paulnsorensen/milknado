from __future__ import annotations

from dataclasses import dataclass

from easy_cheese_schemas import SourcePlanRef

from milknado.domains.batching import BatchPlan
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import MilknadoPlanV2, apply_batches_to_graph


@dataclass(frozen=True)
class AppliedMilknadoPlan:
    source_plan_ref: SourcePlanRef
    batch_plan: BatchPlan
    task_node_ids: tuple[int, ...]
    goal_node_id: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.source_plan_ref, SourcePlanRef):
            raise ValueError("applied plan must contain a SourcePlanRef")
        if not isinstance(self.batch_plan, BatchPlan):
            raise ValueError("applied plan must contain a BatchPlan")
        if not isinstance(self.task_node_ids, tuple):
            raise ValueError("task_node_ids must be a tuple")
        if len(self.task_node_ids) != len(self.batch_plan.batches):
            raise ValueError("task node count must match batch count")
        if any(type(node_id) is not int or node_id <= 0 for node_id in self.task_node_ids):
            raise ValueError("task node IDs must be positive integers")
        if len(set(self.task_node_ids)) != len(self.task_node_ids):
            raise ValueError("task node IDs must be unique")
        if self.goal_node_id is not None and (
            type(self.goal_node_id) is not int or self.goal_node_id <= 0
        ):
            raise ValueError("goal_node_id must be a positive integer or None")


def _validate_batch_coverage(plan: MilknadoPlanV2, batch_plan: BatchPlan) -> None:
    expected = tuple(item.change.id for item in plan.items)
    actual = tuple(change_id for batch in batch_plan.batches for change_id in batch.change_ids)
    if len(set(actual)) != len(actual):
        raise ValueError("batch plan change IDs must be unique")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        raise ValueError(
            f"batch/plan change coverage mismatch: missing={missing!r}, unknown={unknown!r}"
        )
    if len({batch.index for batch in batch_plan.batches}) != len(batch_plan.batches):
        raise ValueError("batch indexes must be unique")


def apply_milknado_plan_to_graph(
    graph: MikadoGraph,
    batch_plan: BatchPlan,
    plan: MilknadoPlanV2,
    *,
    parent_id: int | None = None,
) -> AppliedMilknadoPlan:
    """Apply an imported plan and retain the exact batch-to-node execution map."""
    if not isinstance(plan, MilknadoPlanV2):
        raise ValueError("plan must be a MilknadoPlanV2")
    if not isinstance(batch_plan, BatchPlan):
        raise ValueError("batch_plan must be a BatchPlan")
    _validate_batch_coverage(plan, batch_plan)

    created = apply_batches_to_graph(graph, batch_plan, plan.manifest, parent_id=parent_id)
    if parent_id is None:
        goal_node_id = created[0]
        task_node_ids = tuple(created[1:])
    else:
        goal_node_id = None
        task_node_ids = tuple(created)
    return AppliedMilknadoPlan(
        source_plan_ref=plan.source_plan_ref,
        batch_plan=batch_plan,
        task_node_ids=task_node_ids,
        goal_node_id=goal_node_id,
    )


__all__ = ["AppliedMilknadoPlan", "apply_milknado_plan_to_graph"]
