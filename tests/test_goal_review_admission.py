from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import NodeKind, NodeSpec, SessionAction
from milknado.domains.graph import (
    GoalAdmissionDenied,
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    GoalReviewRequest,
    GoalReviewSubjectError,
    GraphCommand,
    MikadoGraph,
)
from milknado.mcp.goal_review import (
    milknado_goal_admission,
    milknado_goal_review_decide,
    milknado_goal_review_request,
)

NOW = "2026-09-13T12:00:00+00:00"
LATER = "2026-09-13T12:05:00+00:00"


def _hierarchy(path: Path, *, project_db: bool = False) -> tuple[MikadoGraph, dict[str, int]]:
    db_path = path / ".milknado" / "milknado.db" if project_db else path / "graph.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    graph = MikadoGraph(db_path)
    root = graph.add_node("project", spec=NodeSpec(kind=NodeKind.GOAL))
    goal_a = graph.add_node("goal a", root.id, NodeSpec(kind=NodeKind.GOAL))
    goal_b = graph.add_node("goal b", root.id, NodeSpec(kind=NodeKind.GOAL))
    task_a1 = graph.add_node("a1", goal_a.id)
    task_a2 = graph.add_node("a2", goal_a.id)
    task_b = graph.add_node("b1", goal_b.id)
    nested = graph.add_node("nested", goal_a.id, NodeSpec(kind=NodeKind.GOAL))
    nested_task = graph.add_node("nested task", nested.id)
    return graph, {
        "root": root.id,
        "goal_a": goal_a.id,
        "goal_b": goal_b.id,
        "a1": task_a1.id,
        "a2": task_a2.id,
        "b1": task_b.id,
        "nested": nested.id,
        "nested_task": nested_task.id,
    }


def _request(
    graph: MikadoGraph, goal_id: int, affected: tuple[int, ...] | None = None
) -> GoalReviewRecord:
    return graph.request_goal_review(
        GoalReviewRequest(
            goal_id=goal_id,
            goal_revision="sha256:goal-contract",
            evidence="New evidence may change the agreed outcome.",
            proposed_change="Change the top-level outcome.",
            affected_node_ids=affected,
            reviewer="worker-1",
            assessed_at=NOW,
        )
    )


def test_review_subject_is_explicit_execution_goal(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        record = _request(graph, nodes["goal_a"], (nodes["a1"],))
        assert record.goal_id == nodes["goal_a"]
        assert record.goal_revision == "sha256:goal-contract"
        assert record.evidence == "New evidence may change the agreed outcome."
        assert record.proposed_change == "Change the top-level outcome."
        assert record.affected_node_ids == (nodes["a1"],)
        assert record.reviewer == "worker-1"
        assert record.decision is GoalReviewDecision.PENDING
        for subject in (nodes["a1"], nodes["nested"]):
            with pytest.raises(GoalReviewSubjectError, match="explicit execution GOAL"):
                _ = _request(graph, subject)
    finally:
        graph.close()


def test_goal_allows_only_one_pending_review(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        _ = _request(graph, nodes["goal_a"])
        with pytest.raises(ValueError, match="already has a pending review"):
            _ = _request(graph, nodes["goal_a"])
    finally:
        graph.close()


def test_bounded_review_pauses_only_affected_work(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        review = _request(graph, nodes["goal_a"], (nodes["a1"],))
        ready = {node.id for node in graph.get_ready_nodes()}
        assert nodes["a1"] not in ready
        assert {nodes["a2"], nodes["b1"]} <= ready
        with pytest.raises(GoalAdmissionDenied, match="execution paused"):
            _ = graph.claim_node(nodes["a1"], "run-a1", now=NOW)
        with pytest.raises(GoalAdmissionDenied, match="execution paused"):
            graph.claim_ancestor_goal_for_dispatch(nodes["a1"], "dispatch-a1", now=NOW)
        assert (
            graph.ancestor_goal_claimed_by_other(nodes["a1"], caller_run_id="different-dispatch")
            is None
        )
        assert graph.claim_node(nodes["a2"], "run-a2", now=NOW)
        assert graph.goal_review_interruption_targets(review.review_id) == ()
    finally:
        graph.close()


def test_ready_limit_is_applied_after_review_exclusion(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        root = graph.add_node("project", spec=NodeSpec(kind=NodeKind.GOAL))
        paused_goal = graph.add_node("paused", root.id, NodeSpec(kind=NodeKind.GOAL))
        for index in range(100):
            _ = graph.add_node(f"paused-{index}", paused_goal.id)
        active_goal = graph.add_node("active", root.id, NodeSpec(kind=NodeKind.GOAL))
        active = graph.add_node("active-task", active_goal.id)
        _ = _request(graph, paused_goal.id)

        assert [node.id for node in graph.get_ready_nodes(limit=1)] == [active.id]
    finally:
        graph.close()


def test_unbounded_review_pauses_one_execution_goal(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        _ = _request(graph, nodes["goal_a"])
        ready = {node.id for node in graph.get_ready_nodes()}
        assert not {nodes["a1"], nodes["a2"], nodes["nested_task"]} & ready
        assert nodes["b1"] in ready
        assert graph.claim_node(nodes["b1"], "run-b1", now=NOW)
    finally:
        graph.close()


@pytest.mark.parametrize("decision", [GoalReviewDecision.ACCEPTED, GoalReviewDecision.REJECTED])
def test_terminal_review_decision_resumes_original_goal(
    tmp_path: Path, decision: GoalReviewDecision
) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        review = _request(graph, nodes["goal_a"])
        decided = graph.decide_goal_review(
            GoalReviewDecisionRequest(review.review_id, decision, "human", LATER)
        )
        assert decided.decision is decision
        assert decided.decided_by == "human"
        assert graph.claim_node(nodes["a1"], f"run-{decision}", now=LATER)
    finally:
        graph.close()


def test_blocked_status_does_not_bypass_pending_review(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        graph.mark_blocked(nodes["a1"])
        _ = _request(graph, nodes["goal_a"], (nodes["a1"],))
        with pytest.raises(GoalAdmissionDenied):
            _ = graph.claim_node(nodes["a1"], "run-blocked", now=NOW)
    finally:
        graph.close()


def test_review_gates_continuation_but_allows_safe_interrupt(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        assert graph.claim_node(nodes["a1"], "run-a1", now=NOW)
        graph.runs.start("run-a1", nodes["a1"], "", NOW, None)
        _ = graph.commands.publish_capabilities(
            "run-a1",
            nodes["a1"],
            "invocation",
            "owner",
            ("steer", "interrupt"),
            (),
            published_at=NOW,
        )
        review = _request(graph, nodes["goal_a"], (nodes["a1"],))
        assert graph.goal_review_interruption_targets(review.review_id) == (nodes["a1"],)
        with pytest.raises(GoalAdmissionDenied):
            _ = graph.commands.admit(_command(nodes["a1"], "steer"), now=NOW)
        receipt = graph.commands.admit(_command(nodes["a1"], "interrupt"), now=NOW)
        assert receipt.status == "queued"
        _ = graph.decide_goal_review(
            GoalReviewDecisionRequest(
                review.review_id, GoalReviewDecision.ACCEPTED, "human", LATER
            )
        )
        assert graph.goal_review_interruption_targets(review.review_id) == ()
    finally:
        graph.close()


def _command(node_id: int, action: SessionAction) -> GraphCommand:
    return GraphCommand(
        command_id=f"command-{action}",
        node_id=node_id,
        run_id="run-a1",
        invocation_id="invocation",
        owner_incarnation="owner",
        action=action,
        expires_at=LATER,
    )


def test_goal_review_mcp_publishes_structured_links_and_admission(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path, project_db=True)
    graph.close()
    root = str(tmp_path)

    requested = milknado_goal_review_request(
        goal_id=nodes["goal_a"],
        goal_revision="revision-7",
        evidence="Evidence changed.",
        proposed_change="Change the outcome.",
        reviewer="worker-7",
        affected_node_ids=[nodes["a1"]],
        project_root=root,
    )

    assert requested["decision"] == "pending"
    assert requested["links"] == {
        "goal": {"kind": "node", "node_id": nodes["goal_a"]},
        "safe_interruption_targets": (),
    }
    assert milknado_goal_admission(nodes["a1"], root)["allowed"] is False
    review_id = requested["review_id"]
    assert isinstance(review_id, int)
    decided = milknado_goal_review_decide(review_id, "accepted", "human-1", root)
    assert decided["decision"] == "accepted"
    assert milknado_goal_admission(nodes["a1"], root)["allowed"] is True


def test_review_on_ancestor_execution_goal_denies_nested_execution_goal_descendant(
    tmp_path: Path,
) -> None:
    # root and goal_a both qualify as "execution goals"; nested_task's nearest
    # execution goal is goal_a, not root. A pending review on the ANCESTOR
    # (root) must still gate nested_task -- goal_admission must not stop at
    # the nearest execution goal only.
    graph, nodes = _hierarchy(tmp_path)
    try:
        _ = _request(graph, nodes["root"])
        with pytest.raises(GoalAdmissionDenied, match="execution paused"):
            _ = graph.claim_node(nodes["nested_task"], "run-nested", now=NOW)
        ready = {node.id for node in graph.get_ready_nodes()}
        assert nodes["nested_task"] not in ready
    finally:
        graph.close()


def test_admit_command_retry_returns_stored_receipt_under_pending_review(
    tmp_path: Path,
) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        assert graph.claim_node(nodes["a1"], "run-a1", now=NOW)
        graph.runs.start("run-a1", nodes["a1"], "", NOW, None)
        _ = graph.commands.publish_capabilities(
            "run-a1", nodes["a1"], "invocation", "owner", ("steer",), (), published_at=NOW
        )
        cmd = _command(nodes["a1"], "steer")
        receipt = graph.commands.admit(cmd, now=NOW)
        assert receipt.status == "queued"

        _ = _request(graph, nodes["goal_a"], (nodes["a1"],))

        retried = graph.commands.admit(cmd, now=NOW)
        assert retried == receipt

        conflicting = GraphCommand(
            command_id=cmd.command_id,
            node_id=nodes["a1"],
            run_id="run-a1",
            invocation_id="invocation",
            owner_incarnation="owner",
            action="steer",
            expires_at=LATER,
            text="different",
        )
        with pytest.raises(ValueError, match="already names a different command"):
            _ = graph.commands.admit(conflicting, now=NOW)
    finally:
        graph.close()
