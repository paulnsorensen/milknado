"""Run-time goal-subtree shape validation for milknado."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from milknado.domains.common import NodeKind, NodeStatus
from milknado.domains.common.types import DEFAULT_FLAVOR
from milknado.domains.graph.status_flow import subtree_post_order

if TYPE_CHECKING:
    from milknado.domains.graph.graph import MikadoGraph


@dataclass
class RunnabilityReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_goal_runnable(graph: MikadoGraph, goal_id: int) -> RunnabilityReport:
    """Validate a goal subtree; malformed topology is reported once and terminates."""
    report = RunnabilityReport()
    node = graph.get_node(goal_id)
    if node is None:
        report.errors.append(f"node {goal_id} not found")
        return report
    if node.kind != NodeKind.GOAL:
        report.errors.append(
            f"node {goal_id} is not a goal (kind={node.kind.value}); "
            + "only goal nodes can be validated for runnability"
        )
        return report
    children_map = graph.get_children_map()
    for current in subtree_post_order(children_map, node):
        children = children_map.get(current.id, [])
        if children:
            continue
        if current.kind == NodeKind.GOAL and current.status == NodeStatus.PENDING:
            report.errors.append(
                f"leaf goal {current.id} has zero children (undecomposed stub); "
                + "decompose into task nodes first"
            )
        elif current.kind == NodeKind.GOAL and current.status != NodeStatus.PENDING:
            continue
        elif current.kind != NodeKind.TASK:
            report.errors.append(
                f"leaf node {current.id} (kind={current.kind.value}) is not a task; "
                + "all leaf descendants must be tasks"
            )
        elif current.flavor == DEFAULT_FLAVOR and not graph.get_file_ownership(current.id):
            report.warnings.append(
                f"task {current.id} has no file hints (implement-flavored tasks "
                + "should declare which files they touch)"
            )
    return report


def invalid_subtree_node_ids(graph: MikadoGraph, goal_id: int) -> set[int]:
    """Return all nodes to skip when a root goal is structurally invalid."""
    goal = graph.get_node(goal_id)
    if goal is None:
        return set()
    return {node.id for node in subtree_post_order(graph.get_children_map(), goal)}


def validate_runnable_roots(
    graph: MikadoGraph,
) -> list[tuple[int, RunnabilityReport]]:
    """Validate every root GOAL and every GOAL directly under a root ROADMAP."""
    results: list[tuple[int, RunnabilityReport]] = []
    for root in graph.get_roots():
        if root.kind == NodeKind.GOAL:
            results.append((root.id, validate_goal_runnable(graph, root.id)))
        elif root.kind == NodeKind.ROADMAP:
            for child in graph.get_children(root.id):
                if child.kind == NodeKind.GOAL:
                    results.append((child.id, validate_goal_runnable(graph, child.id)))
    return results
