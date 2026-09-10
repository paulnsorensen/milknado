from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from easy_cheese_schemas import (
    CurdPlan,
    canonical_bytes,
    canonical_digest,
    validate_curd_plan,
)

from milknado.domains.common import NodeKind, NodeSpec

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True, slots=True)
class CurdAdmission:
    goal_id: int
    curd_node_ids: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "curd_node_ids",
            MappingProxyType(dict(self.curd_node_ids)),
        )


def admit_curd_plan(
    plan: CurdPlan,
    graph: MikadoGraph,
    *,
    parent_id: int | None = None,
) -> CurdAdmission:
    """Admit one validated semantic plan into the Mikado graph.

    A plan without a containing parent creates one GOAL.  When a parent is
    supplied, it is the existing containing node and no synthetic goal is
    added.  Curd dependencies are graph edges from the dependent task to its
    prerequisite, matching Mikado's readiness traversal.
    """
    plan = validate_curd_plan(plan)
    payload = canonical_bytes(plan)
    if parent_id is None:
        goal_id = graph.add_node(
            plan.objective,
            spec=NodeSpec(kind=NodeKind.GOAL),
        ).id
    else:
        if graph.get_node(parent_id) is None:
            raise ValueError(f"parent_id {parent_id} not found")
        goal_id = parent_id

    node_ids: dict[str, int] = {}
    curd_digests: dict[str, str] = {}
    for curd in plan.curds:
        node = graph.add_node(
            curd.outcome,
            parent_id=goal_id,
            spec=NodeSpec(kind=NodeKind.TASK),
        )
        graph.files.claim(node.id, list(curd.scope.paths))
        node_ids[curd.curd_id] = node.id
        curd_digests[curd.curd_id] = canonical_digest(curd)

    for curd in plan.curds:
        for dependency in curd.dependencies:
            _ = graph.add_edge(node_ids[curd.curd_id], node_ids[dependency])

    graph.record_curd_plan(plan.plan_id, plan.revision, plan.digest, payload)
    for curd in plan.curds:
        graph.record_curd_node(
            node_ids[curd.curd_id],
            plan.plan_id,
            plan.revision,
            curd.curd_id,
            curd_digests[curd.curd_id],
        )
    return CurdAdmission(goal_id=goal_id, curd_node_ids=node_ids)


__all__ = ["CurdAdmission", "admit_curd_plan"]
