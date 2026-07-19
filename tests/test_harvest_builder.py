"""`build_harvest_summary`: the single format-neutral rollup shared by the wiki
and github exporters (Acceptance 7).

Wiki markdown byte-identity is already pinned by test_wiki_exporter.py; here we
pin the summary's own fields — done/failed counts, result-deposit summaries, and
deduped sorted branch names — directly.
"""

from __future__ import annotations

from milknado.domains.common import NodeKind, NodeSpec, RunResult
from milknado.domains.common.harvest import (
    _bounded_results,
    _truncate_result,
    build_harvest_summary,
)
from milknado.domains.graph import MikadoGraph

NOW = "2026-06-08T12:00:00Z"


def _goal(graph: MikadoGraph) -> int:
    roadmap = graph.add_node("rm", spec=NodeSpec(kind=NodeKind.ROADMAP))
    goal = graph.add_node("g", parent_id=roadmap.id, spec=NodeSpec(kind=NodeKind.GOAL))
    return goal.id


def test_status_reflects_goal_status(graph: MikadoGraph) -> None:
    goal_id = _goal(graph)
    graph.mark_running(goal_id)
    graph.mark_done(goal_id)
    goal = graph.get_node(goal_id)
    assert goal is not None
    assert build_harvest_summary(graph, goal).status == "done"


def test_counts_done_and_failed_tasks(graph: MikadoGraph) -> None:
    goal_id = _goal(graph)
    t1 = graph.add_node("t1", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
    t2 = graph.add_node("t2", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
    graph.add_node("t3", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
    graph.mark_running(t1.id)
    graph.mark_done(t1.id)
    graph.mark_failed(t2.id)
    # t3 stays pending — counted in neither.
    goal = graph.get_node(goal_id)
    assert goal is not None
    summary = build_harvest_summary(graph, goal)
    assert summary.tasks_done == 1
    assert summary.tasks_failed == 1


def test_nested_tasks_are_walked(graph: MikadoGraph) -> None:
    goal_id = _goal(graph)
    parent = graph.add_node("t1", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
    child = graph.add_node("t1a", parent_id=parent.id, spec=NodeSpec(kind=NodeKind.TASK))
    for tid in (parent.id, child.id):
        graph.mark_running(tid)
        graph.mark_done(tid)
    goal = graph.get_node(goal_id)
    assert goal is not None
    assert build_harvest_summary(graph, goal).tasks_done == 2


def test_result_summaries_collected_from_terminal_runs(graph: MikadoGraph) -> None:
    goal_id = _goal(graph)
    task = graph.add_node("t1", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
    graph.start_run("run-1", task.id, "/tmp/log", NOW, None)
    graph.finish_run("run-1", RunResult(status="done", exit_code=0, timed_out=False, ended_at=NOW))
    graph.deposit_run_message("run-1", "result", "shipped the thing", NOW)
    goal = graph.get_node(goal_id)
    assert goal is not None
    assert build_harvest_summary(graph, goal).result_summaries == ["shipped the thing"]


def test_result_summaries_are_bounded(graph: MikadoGraph) -> None:
    goal_id = _goal(graph)
    task = graph.add_node("t1", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
    graph.start_run("run-large", task.id, "/tmp/log", NOW, None)
    graph.finish_run(
        "run-large",
        RunResult(status="done", exit_code=0, timed_out=False, ended_at=NOW),
    )
    graph.deposit_run_message("run-large", "result", "x" * 10_000, NOW)
    goal = graph.get_node(goal_id)
    assert goal is not None

    summaries = build_harvest_summary(graph, goal).result_summaries

    assert len(summaries) == 1
    assert len(summaries[0]) == 4_000
    assert summaries[0].endswith("\n[truncated]")


def test_tiny_result_limit_is_still_bounded() -> None:
    assert _truncate_result("payload", 3) == "\n[t"


def test_total_result_payload_is_bounded() -> None:
    summaries = _bounded_results(["x" * 4_000] * 9)

    assert len(summaries) == 8
    assert sum(map(len, summaries)) == 32_000


def test_branch_names_sorted_and_deduped(graph: MikadoGraph) -> None:
    goal_id = _goal(graph)
    t1 = graph.add_node("t1", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
    t2 = graph.add_node("t2", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK))
    graph.mark_running(t1.id, worktree_path="/wt1", branch_name="claude/b", run_id="r1")
    graph.mark_running(t2.id, worktree_path="/wt2", branch_name="claude/a", run_id="r2")
    goal = graph.get_node(goal_id)
    assert goal is not None
    assert build_harvest_summary(graph, goal).branch_names == ["claude/a", "claude/b"]


def test_empty_goal_has_zero_rollup(graph: MikadoGraph) -> None:
    goal_id = _goal(graph)
    goal = graph.get_node(goal_id)
    assert goal is not None
    summary = build_harvest_summary(graph, goal)
    assert summary.tasks_done == 0
    assert summary.tasks_failed == 0
    assert summary.result_summaries == []
    assert summary.branch_names == []
