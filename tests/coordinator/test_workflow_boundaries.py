from pathlib import Path
from typing import cast

import pytest

from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.execution import NodeLoopOutcome
from milknado.domains.graph import GoalReviewRequest, GroupWorkspace, MikadoGraph


def test_group_and_revision_rejections_preserve_links_and_events(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    conn = graph.group_connection
    workflow = CoordinatorWorkflow(graph, conn)
    owner = workflow.start_goal("Owner", "codex")
    other = workflow.start_goal("Other", "codex")
    foreign = graph.add_node("Foreign task", other.goal_id)
    local = graph.add_node("Local task", owner.goal_id)
    workspace = GroupWorkspace(str(tmp_path / "group"), "branch", "provider")
    links = conn.execute("SELECT * FROM coordinator_links").fetchall()
    events = conn.execute("SELECT * FROM coordinator_events").fetchall()

    with pytest.raises(ValueError, match="another goal"):
        _ = workflow.create_group(owner, "main", (foreign.id,), workspace)
    with pytest.raises(ValueError, match="revision identity"):
        workflow.record_revision(owner, "", (local.id,))
    with pytest.raises(ValueError, match="not admitted"):
        workflow.record_revision(owner, "revision", (foreign.id,))
    with pytest.raises(ValueError, match="another coordinator goal"):
        _ = workflow.review_goal_change(
            owner,
            GoalReviewRequest(
                other.goal_id,
                "revision",
                "evidence",
                "change",
                (foreign.id,),
                "agent",
                operation_id="review",
            ),
        )
    assert conn.execute("SELECT * FROM coordinator_links").fetchall() == links
    assert conn.execute("SELECT * FROM coordinator_events").fetchall() == events
    assert conn.execute("SELECT COUNT(*) FROM execution_groups").fetchone()[0] == 0
    graph.close()


def test_conflicting_finish_preserves_task_result_and_dispatch(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    conn = graph.group_connection
    workflow = CoordinatorWorkflow(graph, conn)
    session = workflow.start_goal("Deliver", "codex")
    task = graph.add_node("Task", session.goal_id)
    group = workflow.create_group(
        session,
        "main",
        (task.id,),
        GroupWorkspace(str(tmp_path / "group"), "branch", "provider"),
    )
    handoff = workflow.dispatch_task(session, group.id, task.id, "run")
    _ = workflow.acknowledge_launch(session, handoff)
    workflow.finish_task(session, handoff.attempt, NodeLoopOutcome(task.id, True, "verified"))
    before = conn.execute("SELECT * FROM coordinator_events").fetchall()
    dispatch = cast(
        tuple[str] | None,
        conn.execute(
            "SELECT state FROM coordinator_dispatches WHERE attempt_id = ?",
            (handoff.attempt.attempt_id,),
        ).fetchone(),
    )

    with pytest.raises(ValueError, match="conflicts"):
        workflow.finish_task(session, handoff.attempt, NodeLoopOutcome(task.id, False, "wrong"))
    assert graph.groups.task_result(task.id) == ("done", "verified")
    assert conn.execute("SELECT * FROM coordinator_events").fetchall() == before
    unchanged = cast(
        tuple[str] | None,
        conn.execute(
            "SELECT state FROM coordinator_dispatches WHERE attempt_id = ?",
            (handoff.attempt.attempt_id,),
        ).fetchone(),
    )
    assert unchanged == dispatch
    graph.close()
