"""Run-time goal-subtree shape validation for milknado."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from milknado.domains.common import NodeKind, NodeStatus
from milknado.domains.common.types import DEFAULT_FLAVOR

if TYPE_CHECKING:
    from milknado.domains.graph.graph import MikadoGraph


@dataclass
class RunnabilityReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_goal_runnable(graph: MikadoGraph, goal_id: int) -> RunnabilityReport:
    """Validate that a goal subtree is ready to run.

    Returns a RunnabilityReport with errors (hard blockers) and warnings
    (advisory only — never block a run).
    """
    report = RunnabilityReport()
    node = graph.get_node(goal_id)
    if node is None:
        report.errors.append(f"node {goal_id} not found")
        return report

    if node.kind != NodeKind.GOAL:
        report.errors.append(
            f"node {goal_id} is not a goal (kind={node.kind.value}); "
            "only goal nodes can be validated for runnability"
        )
        return report

    children_map = graph.get_children_map()
    _check_subtree(graph, goal_id, children_map, report)
    return report


def _check_subtree(
    graph: MikadoGraph,
    node_id: int,
    children_map: dict,
    report: RunnabilityReport,
) -> None:
    node = graph.get_node(node_id)
    if node is None:
        return

    children = children_map.get(node_id, [])
    is_leaf = not children

    if is_leaf:
        if node.kind == NodeKind.GOAL and node.status == NodeStatus.PENDING:
            report.errors.append(
                f"leaf goal {node_id} has zero children (undecomposed stub); "
                "decompose into task nodes first"
            )
        elif node.kind == NodeKind.GOAL and node.status != NodeStatus.PENDING:
            pass  # DONE/FAILED/BLOCKED leaf goal: not an undecomposed stub
        elif node.kind != NodeKind.TASK:
            report.errors.append(
                f"leaf node {node_id} (kind={node.kind.value}) is not a task; "
                "all leaf descendants must be tasks"
            )
        else:
            # Advisory warning: implement-flavored task without file hints
            if node.flavor == DEFAULT_FLAVOR:
                files = graph.get_file_ownership(node_id)
                if not files:
                    report.warnings.append(
                        f"task {node_id} has no file hints (implement-flavored tasks "
                        "should declare which files they touch)"
                    )

    for child in children:
        _check_subtree(graph, child.id, children_map, report)


def validate_runnable_roots(
    graph: MikadoGraph,
) -> list[tuple[int, RunnabilityReport]]:
    """Validate every root GOAL (and every GOAL child of a root ROADMAP).

    Returns a list of (goal_id, report) pairs. Only GOALs are validated;
    bare TASK roots are skipped (legal per spec). ROADMAP roots are skipped
    at the root level but their GOAL children are validated.
    """
    results: list[tuple[int, RunnabilityReport]] = []
    for root in graph.get_roots():
        if root.kind == NodeKind.GOAL:
            results.append((root.id, validate_goal_runnable(graph, root.id)))
        elif root.kind == NodeKind.ROADMAP:
            for child in graph.get_children(root.id):
                if child.kind == NodeKind.GOAL:
                    results.append((child.id, validate_goal_runnable(graph, child.id)))
    return results
