from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import NodeStatus
from milknado.domains.graph import (
    ConcurrencyLimitReached,
    GroupWorkspace,
    MikadoGraph,
    TaskOutcome,
)


def _workspace(name: str) -> GroupWorkspace:
    return GroupWorkspace(f"/tmp/{name}", name, f"session-{name}")


def test_external_prerequisite_is_required_at_admission_and_start(graph: MikadoGraph) -> None:
    prerequisite = graph.add_node("prerequisite")
    task = graph.add_node("task")
    _ = graph.add_edge(task.id, prerequisite.id)
    with pytest.raises(ValueError, match="external prerequisite"):
        _ = graph.groups.create("graph-a", (task.id,), _workspace("one"))

    graph.mark_running(prerequisite.id)
    graph.mark_done(prerequisite.id)
    group = graph.groups.create("graph-a", (task.id,), _workspace("one"))
    late_prerequisite = graph.add_node("late prerequisite")
    _ = graph.add_edge(task.id, late_prerequisite.id)
    with pytest.raises(ValueError, match="external prerequisite"):
        _ = graph.groups.start_task(group.id, task.id, "run-a")


def test_fork_clones_tasks_into_durable_alternative(graph: MikadoGraph) -> None:
    source_task = graph.add_node("source task", files=("a.py",))
    source = graph.groups.create("graph-a", (source_task.id,), _workspace("source"))
    fork = graph.groups.fork(source.id, _workspace("fork"))
    (fork_task_id,) = graph.groups.tasks(fork.id)

    assert fork.graph_id != source.graph_id
    assert fork_task_id != source_task.id
    fork_task = graph.get_node(fork_task_id)
    assert fork_task is not None
    assert fork_task.description == "source task"
    assert graph.files.for_node(fork_task_id) == ["a.py"]
    assert graph.groups.tasks(source.id) == (source_task.id,)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        relation = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT source_group_id FROM graph_alternatives WHERE id = ?", (fork.graph_id,)
            ).fetchone(),
        )
    assert relation == (source.id,)


def test_group_rejects_task_already_claimed_by_ordinary_run(graph: MikadoGraph) -> None:
    task = graph.add_node("task")
    assert graph.claim_node(task.id, "ordinary", now="2026-10-04T00:00:00Z")
    with pytest.raises(ValueError, match="active"):
        _ = graph.groups.create("graph-a", (task.id,), _workspace("one"))


def test_group_claim_uses_canonical_status_and_blocks_ordinary_claim(graph: MikadoGraph) -> None:
    first = graph.add_node("first")
    second = graph.add_node("second")
    group = graph.groups.create("graph-a", (first.id, second.id), _workspace("one"))
    attempt = graph.groups.start_task(group.id, first.id, "run-a")
    running = graph.get_node(first.id)
    assert running is not None
    assert running.status is NodeStatus.RUNNING
    with pytest.raises((sqlite3.IntegrityError, ValueError), match="group writer"):
        _ = graph.claim_node(second.id, "ordinary", now="2026-10-04T00:00:00Z")
    graph.groups.finish_task(attempt, TaskOutcome("done", "first result"))
    done = graph.get_node(first.id)
    assert done is not None
    assert done.status is NodeStatus.DONE
    assert graph.groups.task_result(first.id) == ("done", "first result")


def test_attempt_identity_blocks_reused_run_id(graph: MikadoGraph) -> None:
    first = graph.add_node("first")
    second = graph.add_node("second")
    group = graph.groups.create("graph-a", (first.id, second.id), _workspace("one"))
    first_attempt = graph.groups.start_task(group.id, first.id, "same-run")
    graph.groups.finish_task(first_attempt, TaskOutcome("done", "first"))
    second_attempt = graph.groups.start_task(group.id, second.id, "same-run")
    assert first_attempt.attempt_id != second_attempt.attempt_id
    with pytest.raises(ValueError, match="fence"):
        graph.groups.finish_task(first_attempt, TaskOutcome("done", "stale"))
    graph.groups.finish_task(second_attempt, TaskOutcome("done", "second"))
    assert graph.groups.task_result(first.id) == ("done", "first")
    assert graph.groups.task_result(second.id) == ("done", "second")


def test_delete_cleans_membership_and_rejects_active_task(graph: MikadoGraph) -> None:
    task = graph.add_node("task")
    group = graph.groups.create("graph-a", (task.id,), _workspace("one"))
    attempt = graph.groups.start_task(group.id, task.id, "run-a")
    with pytest.raises(ValueError, match="active execution group"):
        _ = graph.delete_node(task.id)
    graph.groups.finish_task(attempt, TaskOutcome("done", "result"))
    assert graph.delete_node(task.id) == 1
    assert graph.groups.tasks(group.id) == ()


def test_fork_preserves_internal_dependency_with_new_task_ids(graph: MikadoGraph) -> None:
    prerequisite = graph.add_node("prerequisite")
    dependent = graph.add_node("dependent")
    _ = graph.add_edge(dependent.id, prerequisite.id)
    source = graph.groups.create("graph-a", (prerequisite.id, dependent.id), _workspace("source"))
    fork = graph.groups.fork(source.id, _workspace("fork"))
    fork_prerequisite, fork_dependent = graph.groups.tasks(fork.id)
    assert {fork_prerequisite, fork_dependent}.isdisjoint({prerequisite.id, dependent.id})
    assert [node.id for node in graph.get_children(fork_dependent)] == [fork_prerequisite]
    assert [node.id for node in graph.get_children(dependent.id)] == [prerequisite.id]


def test_fork_rejects_source_without_tasks(graph: MikadoGraph) -> None:
    task = graph.add_node("task")
    source = graph.groups.create("graph-a", (task.id,), _workspace("source"))
    assert graph.delete_node(task.id) == 1
    with pytest.raises(ValueError, match="no tasks"):
        _ = graph.groups.fork(source.id, _workspace("fork"))


def test_capacity_failure_clears_group_writer(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "capacity.db", concurrency_limit=1)
    try:
        occupied = graph.add_node("occupied")
        task = graph.add_node("task")
        group = graph.groups.create("graph-a", (task.id,), _workspace("one"))
        assert graph.claim_node(occupied.id, "occupied-run", now="2026-10-04T00:00:00Z")
        with pytest.raises(ConcurrencyLimitReached):
            _ = graph.groups.start_task(group.id, task.id, "group-run")
        with closing(sqlite3.connect(graph.db_path)) as conn:
            writer = cast(
                tuple[None] | None,
                conn.execute(
                    "SELECT active_attempt_id FROM execution_groups WHERE id = ?", (group.id,)
                ).fetchone(),
            )
        assert writer == (None,)
    finally:
        graph.close()


def test_blocked_result_updates_canonical_status(graph: MikadoGraph) -> None:
    task = graph.add_node("task")
    group = graph.groups.create("graph-a", (task.id,), _workspace("one"))
    attempt = graph.groups.start_task(group.id, task.id, "run-a")
    graph.groups.finish_task(attempt, TaskOutcome("blocked", "needs input"))
    node = graph.get_node(task.id)
    assert node is not None
    assert node.status is NodeStatus.BLOCKED
    assert graph.groups.task_result(task.id) == ("blocked", "needs input")


def test_lost_canonical_fence_cannot_complete_group(graph: MikadoGraph) -> None:
    task = graph.add_node("task")
    group = graph.groups.create("graph-a", (task.id,), _workspace("one"))
    attempt = graph.groups.start_task(group.id, task.id, "run-a")
    assert graph.release(task.id, attempt.attempt_id)
    with pytest.raises(ValueError, match="fence"):
        graph.groups.finish_task(attempt, TaskOutcome("done", "wrong"))
    assert graph.groups.task_result(task.id) is None


def test_terminal_group_write_rolls_back_canonical_status(graph: MikadoGraph) -> None:
    task = graph.add_node("task")
    group = graph.groups.create("graph-a", (task.id,), _workspace("one"))
    attempt = graph.groups.start_task(group.id, task.id, "run-a")
    with closing(sqlite3.connect(graph.db_path)) as conn, conn:
        _ = conn.execute(
            "CREATE TRIGGER fail_group_result BEFORE UPDATE OF status ON execution_group_tasks "
            + "BEGIN SELECT RAISE(ABORT, 'result write interrupted'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="result write interrupted"):
        graph.groups.finish_task(attempt, TaskOutcome("done", "result"))
    node = graph.get_node(task.id)
    assert node is not None
    assert node.status is NodeStatus.RUNNING
    assert graph.groups.task_result(task.id) is None
    with closing(sqlite3.connect(graph.db_path)) as conn:
        writer = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT active_attempt_id FROM execution_groups WHERE id = ?", (group.id,)
            ).fetchone(),
        )
    assert writer == (attempt.attempt_id,)


def test_fork_remaps_internal_containment_and_detaches_external_parent(
    graph: MikadoGraph,
) -> None:
    outside = graph.add_node("outside")
    parent = graph.add_node("parent", parent_id=outside.id)
    child = graph.add_node("child", parent_id=parent.id)
    graph.mark_running(outside.id)
    graph.mark_done(outside.id)
    source = graph.groups.create("graph-a", (child.id, parent.id), _workspace("source"))
    fork = graph.groups.fork(source.id, _workspace("fork"))
    fork_child, fork_parent = graph.groups.tasks(fork.id)
    copied_parent = graph.get_node(fork_parent)
    copied_child = graph.get_node(fork_child)
    assert copied_parent is not None and copied_child is not None
    assert copied_parent.parent_id is None
    assert copied_child.parent_id == fork_parent
    assert graph.delete_node(outside.id, cascade=True) == 3
    assert graph.get_node(fork_parent) is not None
    assert graph.get_node(fork_child) is not None


def test_late_internal_dependency_must_match_group_order(graph: MikadoGraph) -> None:
    first = graph.add_node("first")
    second = graph.add_node("second")
    group = graph.groups.create("graph-a", (first.id, second.id), _workspace("one"))
    _ = graph.add_edge(first.id, second.id)
    with pytest.raises(ValueError, match="order"):
        _ = graph.groups.start_task(group.id, first.id, "run-a")
    node = graph.get_node(first.id)
    assert node is not None
    assert node.status is NodeStatus.PENDING
