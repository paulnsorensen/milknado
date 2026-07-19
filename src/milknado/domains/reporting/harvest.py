"""Shared harvest rollup: fold a goal's task subtree into an execution summary.

Both the wiki exporter and the github exporter project the same outcome — task
done/failed counts, result-message summaries, and branch names. This builder is
the single source of that rollup, and `format_harvest_text` is the single source
of its plain-text rendering, so the two projections never drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph


MAX_HARVEST_RESULTS = 100
MAX_HARVEST_RESULT_CHARS = 4_000
MAX_HARVEST_TOTAL_CHARS = 32_000


def _truncate_result(payload: str, limit: int) -> str:
    marker = "\n[truncated]"
    if len(payload) <= limit:
        return payload
    if limit <= len(marker):
        return marker[:limit]
    return f"{payload[: limit - len(marker)]}{marker}"


def _bounded_results(results: list[str]) -> list[str]:
    bounded: list[str] = []
    remaining = MAX_HARVEST_TOTAL_CHARS
    for payload in results[:MAX_HARVEST_RESULTS]:
        if remaining <= 0:
            break
        result = _truncate_result(payload, min(MAX_HARVEST_RESULT_CHARS, remaining))
        bounded.append(result)
        remaining -= len(result)
    return bounded


@dataclass(frozen=True)
class HarvestSummary:
    status: str  # the goal node's own status value
    tasks_done: int
    tasks_failed: int
    result_summaries: list[str]  # latest "result" deposit per terminal task run
    branch_names: list[str]  # sorted, deduped branch names across the subtree


def _task_descendants(graph: MikadoGraph, goal_id: int) -> list[MikadoNode]:
    children: dict[int, list[int]] = {}
    nodes = graph.get_all_nodes()
    for node in nodes:
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node.id)
    task_ids: list[int] = []
    stack = list(children.get(goal_id, ()))
    while stack:
        node_id = stack.pop()
        task_ids.append(node_id)
        stack.extend(children.get(node_id, ()))
    return [node for node in graph.get_nodes(task_ids) if node.kind == NodeKind.TASK]


def _collect_summaries(graph: MikadoGraph, tasks: list[MikadoNode]) -> list[str]:
    results = graph.latest_results_for_nodes(task.id for task in tasks)
    return _bounded_results([results[task.id].strip() for task in tasks if task.id in results])


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


def format_harvest_text(summary: HarvestSummary) -> str:
    """Render a harvest summary as plain text.

    No separate `follow-ups:` line (despite the spec's illustrated schema):
    milknado tracks follow-ups as new graph nodes, not a queryable field, so
    they already fold into the task done/failed rollup below.
    """
    lines = [
        f"result: {summary.status} · tasks: "
        f"{summary.tasks_done} done / {summary.tasks_failed} failed"
    ]
    if summary.result_summaries:
        lines.append("summary: " + " | ".join(summary.result_summaries))
    if summary.branch_names:
        lines.append("branches: " + ", ".join(summary.branch_names))
    return "\n".join(lines)
