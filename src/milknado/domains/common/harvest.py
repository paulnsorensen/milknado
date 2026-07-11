"""Shared harvest rollup: fold a goal's task subtree into an execution summary.

Both the wiki exporter and the github exporter project the same outcome — task
done/failed counts, result-message summaries, and branch names — into their own
target format. This builder is the single source of that rollup so the two
projections never drift; each caller formats the returned summary its own way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from milknado.domains.common.types import MikadoNode, NodeKind, NodeStatus

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class HarvestSummary:
    status: str  # the goal node's own status value
    tasks_done: int
    tasks_failed: int
    result_summaries: list[str]  # latest "result" deposit per terminal task run
    branch_names: list[str]  # sorted, deduped branch names across the subtree


def _task_descendants(graph: MikadoGraph, goal_id: int) -> list[MikadoNode]:
    """Collect TASK nodes in the goal's subtree, not crossing into sibling goals."""
    tasks: list[MikadoNode] = []
    stack = list(graph.get_children(goal_id))
    while stack:
        node = stack.pop()
        if node.kind == NodeKind.TASK:
            tasks.append(node)
            stack.extend(graph.get_children(node.id))
    return tasks


def _collect_summaries(graph: MikadoGraph, tasks: list[MikadoNode]) -> list[str]:
    summaries: list[str] = []
    for task in tasks:
        for run in graph.runs_for_node(task.id, terminal_only=True):
            msg = graph.latest_run_message(run["run_id"], "result")
            if msg:
                summaries.append(msg.strip())
    return summaries


def build_harvest_summary(graph: MikadoGraph, goal: MikadoNode) -> HarvestSummary:
    """Roll a goal's task subtree into a format-neutral execution summary."""
    tasks = _task_descendants(graph, goal.id)
    done = sum(1 for t in tasks if t.status == NodeStatus.DONE)
    failed = sum(1 for t in tasks if t.status == NodeStatus.FAILED)
    branches = sorted({t.branch_name for t in tasks if t.branch_name})
    return HarvestSummary(
        status=goal.status.value,
        tasks_done=done,
        tasks_failed=failed,
        result_summaries=_collect_summaries(graph, tasks),
        branch_names=branches,
    )
