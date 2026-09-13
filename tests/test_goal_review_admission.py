from __future__ import annotations

import getpass
import sqlite3
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast

import pytest
import typer
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.common import (
    CONTROLLER_MASTER_ENV,
    NodeKind,
    NodeSpec,
    NodeStatus,
    SessionAction,
)
from milknado.domains.graph import (
    GoalAdmissionDenied,
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    GoalReviewRequest,
    GoalReviewSubjectError,
    GraphCommand,
    MikadoGraph,
    OwnerCapabilities,
)
from milknado.domains.graph._command_records import get_capabilities
from milknado.domains.graph._pipeline import StatusPipeline
from milknado.mcp.goal_review import (
    milknado_goal_admission,
    milknado_goal_review_request,
)
from tests.graph_helpers import graph_conn

NOW = "2026-09-13T12:00:00+00:00"
LATER = "2026-09-13T12:05:00+00:00"
cli_runner = CliRunner()


def _confirm(*_args: object, **_kwargs: object) -> bool:
    return True


def _register_controller(graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
    graph.register_controller_master()


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
        with pytest.raises(GoalAdmissionDenied):
            graph.mark_running(nodes["a1"])
        assert {nodes["a2"], nodes["b1"]} <= ready
        with pytest.raises(GoalAdmissionDenied):
            _ = graph.set_todo_status(nodes["a1"], NodeStatus.RUNNING)
        with pytest.raises(GoalAdmissionDenied):
            _ = graph.set_subtree_status(nodes["goal_a"], NodeStatus.RUNNING)
        unaffected = graph.get_node(nodes["a2"])
        assert unaffected is not None
        assert unaffected.status is NodeStatus.PENDING
        with pytest.raises(GoalAdmissionDenied, match="execution paused"):
            _ = graph.claim_node(nodes["a1"], "run-a1", now=NOW)
        with pytest.raises(GoalAdmissionDenied, match="execution paused"):
            graph.claim_ancestor_goal_for_dispatch(nodes["a1"], "dispatch-a1", now=NOW)
        assert (
            graph.ancestor_goal_claimed_by_other(nodes["a1"], caller_run_id="different-dispatch")
            is None
        )
        assert graph.claim_node(nodes["a2"], "run-a2", now=NOW)
        assert review.interruption_receipts == ()
    finally:
        graph.close()


def test_bounded_review_expands_nonleaf_affected_scope(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        _ = _request(graph, nodes["goal_a"], (nodes["nested"],))
        assert graph.goal_admission(nodes["nested"]).allowed is False
        assert graph.goal_admission(nodes["nested_task"]).allowed is False
        ready = {node.id for node in graph.get_ready_nodes()}
        assert nodes["a1"] in ready
        assert nodes["nested_task"] not in ready
        with pytest.raises(GoalAdmissionDenied):
            _ = graph.claim_node(nodes["nested_task"], "run-nested", now=NOW)
        assert graph.claim_node(nodes["a1"], "run-a1", now=NOW)
    finally:
        graph.close()


def test_review_wins_direct_claim_race_across_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, nodes = _hierarchy(tmp_path)
    contender = MikadoGraph(tmp_path / "graph.db")
    entered, release = Event(), Event()
    result: list[object] = []
    import milknado.domains.graph._status as status

    original = status.claim_node

    def delayed(  # noqa: PLR0913 - exact monkeypatched claim signature
        pipeline: StatusPipeline,
        conn: sqlite3.Connection,
        node_id: int,
        run_id: str,
        *,
        now: str,
        pid: int | None = None,
    ) -> bool:
        entered.set()
        assert release.wait(5)
        return original(pipeline, conn, node_id, run_id, now=now, pid=pid)

    monkeypatch.setattr(status, "claim_node", delayed)
    worker = Thread(
        target=lambda: result.append(graph.claim_node(nodes["a1"], "run-race", now=NOW))
    )
    worker.start()
    assert entered.wait(5)
    _ = _request(contender, nodes["goal_a"], (nodes["a1"],))
    release.set()
    worker.join(5)
    try:
        assert result == [False]
        node = graph.get_node(nodes["a1"])
        assert node is not None
        assert node.status.value == "pending"
    finally:
        contender.close()
        graph.close()


def test_review_wins_dispatch_goal_claim_race_across_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, nodes = _hierarchy(tmp_path)
    contender = MikadoGraph(tmp_path / "graph.db")
    entered, release = Event(), Event()
    result: list[object] = []
    import milknado.domains.graph._goal_claims as goal_claims

    original = goal_claims.claim_goal_row

    def delayed(  # noqa: PLR0913 - exact monkeypatched claim signature
        conn: goal_claims._ClaimConn,  # pyright: ignore[reportPrivateUsage]
        goal_id: int,
        run_id: str,
        now: str,
        *,
        pid: int | None,
        admission_node_id: int | None = None,
    ) -> bool:
        entered.set()
        assert release.wait(5)
        return original(
            conn,
            goal_id,
            run_id,
            now,
            pid=pid,
            admission_node_id=admission_node_id,
        )

    monkeypatch.setattr(goal_claims, "claim_goal_row", delayed)
    worker = Thread(
        target=lambda: _capture(
            result,
            lambda: graph.claim_ancestor_goal_for_dispatch(
                nodes["a1"], "dispatch-race", now=NOW, pid=123
            ),
        )
    )
    worker.start()
    assert entered.wait(5)
    _ = _request(contender, nodes["goal_a"], (nodes["a1"],))
    release.set()
    worker.join(5)
    try:
        assert len(result) == 1
        assert isinstance(result[0], GoalAdmissionDenied)
    finally:
        contender.close()
        graph.close()


def _capture(results: list[object], operation: Callable[[], object]) -> None:
    try:
        results.append(operation())
    except BaseException as exc:
        results.append(exc)


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
    tmp_path: Path, decision: GoalReviewDecision, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, nodes = _hierarchy(tmp_path)
    _register_controller(graph, monkeypatch)
    try:
        review = _request(graph, nodes["goal_a"])
        decided = graph.decide_goal_review(
            GoalReviewDecisionRequest(review.review_id, decision, LATER),
            decided_by="human",
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


def test_review_gates_continuation_but_allows_safe_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, nodes = _hierarchy(tmp_path)
    _register_controller(graph, monkeypatch)
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
        assert [(item.node_id, item.action) for item in review.interruption_receipts] == [
            (nodes["a1"], "interrupt")
        ]
        with pytest.raises(GoalAdmissionDenied):
            _ = graph.commands.admit(_command(nodes["a1"], "steer"), now=NOW)
        receipt = graph.commands.admit(_command(nodes["a1"], "interrupt"), now=NOW)
        assert receipt.status == "queued"
        _ = graph.decide_goal_review(
            GoalReviewDecisionRequest(review.review_id, GoalReviewDecision.ACCEPTED, LATER),
            decided_by="human",
        )
    finally:
        graph.close()


def test_terminal_capabilities_do_not_requeue_delivered_review_interrupt(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        assert graph.claim_node(nodes["a1"], "run-a1", now=NOW)
        graph.runs.start("run-a1", nodes["a1"], "", NOW, None)
        _ = graph.commands.publish_capabilities(
            "run-a1", nodes["a1"], "invocation", "owner", ("interrupt",), published_at=NOW
        )
        _ = _request(graph, nodes["goal_a"], (nodes["a1"],))
        claimed = graph.commands.claim_pending("run-a1", "owner", now=NOW)
        assert len(claimed) == 1
        _ = graph.commands.publish_capabilities(
            "run-a1", nodes["a1"], "invocation", "owner", ("interrupt",), published_at=LATER
        )
        count_row = cast(
            tuple[int] | None,
            graph_conn(graph).execute("SELECT COUNT(*) FROM session_commands").fetchone(),
        )
        assert count_row is not None
        assert count_row[0] == 1

        terminal = graph.commands.publish_capabilities(
            "run-a1", nodes["a1"], "invocation", "owner", (), published_at=LATER
        )

        assert terminal.actions == ()
    finally:
        graph.close()


def test_pending_review_blocks_approval_but_allows_deny_and_interrupt(
    tmp_path: Path,
) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        assert graph.claim_node(nodes["a1"], "run-a1", now=NOW)
        graph.runs.start("run-a1", nodes["a1"], "", NOW, None)
        _ = graph.commands.publish_capabilities(
            "run-a1",
            nodes["a1"],
            "invocation",
            "owner",
            ("approve", "deny", "interrupt"),
            ("permission-1",),
            published_at=NOW,
        )
        review = _request(graph, nodes["goal_a"], (nodes["a1"],))
        assert len(review.interruption_receipts) == 1
        with pytest.raises(GoalAdmissionDenied):
            _ = graph.commands.admit(_command(nodes["a1"], "approve", "permission-1"), now=NOW)
        assert (
            graph.commands.admit(_command(nodes["a1"], "deny", "permission-1"), now=NOW).status
            == "queued"
        )
        assert graph.commands.admit(_command(nodes["a1"], "interrupt"), now=NOW).status == "queued"
    finally:
        graph.close()


def test_review_request_rolls_back_when_interrupt_inbox_is_full(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path)
    try:
        assert graph.claim_node(nodes["a1"], "run-a1", now=NOW)
        graph.runs.start("run-a1", nodes["a1"], "", NOW, None)
        _ = graph.commands.publish_capabilities(
            "run-a1", nodes["a1"], "invocation", "owner", ("interrupt",), published_at=NOW
        )
        for index in range(64):
            receipt = graph.commands.admit(
                _command(nodes["a1"], "interrupt", command_id=f"filler-{index}"), now=NOW
            )
            assert receipt.status == "queued"

        with pytest.raises(ValueError, match="command inbox is full"):
            _ = _request(graph, nodes["goal_a"], (nodes["a1"],))

        assert graph.get_goal_review(1) is None
        count_row = cast(
            tuple[int] | None,
            graph_conn(graph).execute("SELECT COUNT(*) FROM session_commands").fetchone(),
        )
        assert count_row is not None
        assert count_row[0] == 64
    finally:
        graph.close()


def test_review_interrupt_admission_serializes_owner_replacement(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, nodes = _hierarchy(tmp_path)
    contender = MikadoGraph(tmp_path / "graph.db")
    release = Event()
    try:
        assert graph.claim_node(nodes["a1"], "run-a1", now=NOW)
        graph.runs.start("run-a1", nodes["a1"], "", NOW, None)
        _ = graph.commands.publish_capabilities(
            "run-a1", nodes["a1"], "invocation", "owner", ("interrupt",), published_at=NOW
        )
        import milknado.domains.graph._review_interrupts as review_interrupts

        entered, replacement_started = Event(), Event()
        original = get_capabilities

        def delayed(conn: sqlite3.Connection, run_id: str) -> OwnerCapabilities | None:
            capabilities = original(conn, run_id)
            entered.set()
            if not release.wait(5):
                raise AssertionError("review admission did not resume")
            return capabilities

        monkeypatch.setattr(review_interrupts, "get_capabilities", delayed)
        request_result: list[object] = []
        worker = Thread(
            target=lambda: _capture(
                request_result, lambda: _request(graph, nodes["goal_a"], (nodes["a1"],))
            )
        )
        worker.start()
        assert entered.wait(5)

        replacement_result: list[object] = []

        def publish_replacement() -> object:
            replacement_started.set()
            return contender.commands.publish_capabilities(
                "run-a1",
                nodes["a1"],
                "invocation-new",
                "owner-new",
                ("interrupt",),
                published_at=NOW,
            )

        replacement = Thread(target=lambda: _capture(replacement_result, publish_replacement))
        replacement.start()
        assert replacement_started.wait(5)
        release.set()
        worker.join(5)
        replacement.join(5)

        assert not worker.is_alive()
        assert not replacement.is_alive()
        assert len(request_result) == 1
        assert isinstance(request_result[0], GoalReviewRecord)
        assert request_result[0].interruption_receipts[0].owner_incarnation == "owner"
        assert len(replacement_result) == 1
        assert not isinstance(replacement_result[0], BaseException)
        capabilities = contender.commands.capabilities("run-a1")
        assert capabilities is not None
        assert capabilities.owner_incarnation == "owner-new"
        queued = contender.commands.pending("run-a1", now=NOW)
        assert any(command.owner_incarnation == "owner-new" for command in queued)
    finally:
        release.set()
        contender.close()
        graph.close()


def _command(
    node_id: int,
    action: SessionAction,
    permission_id: str | None = None,
    *,
    command_id: str | None = None,
) -> GraphCommand:
    return GraphCommand(
        command_id=command_id or f"command-{action}",
        node_id=node_id,
        run_id="run-a1",
        invocation_id="invocation",
        owner_incarnation="owner",
        action=action,
        permission_id=permission_id,
        expires_at=LATER,
    )


def test_goal_review_mcp_publishes_receipts_and_admission(tmp_path: Path) -> None:
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
    assert requested["interrupt_receipts"] == ()
    assert milknado_goal_admission(nodes["a1"], root)["allowed"] is False


def test_goal_review_cli_refuses_noninteractive_decision(tmp_path: Path) -> None:
    graph, nodes = _hierarchy(tmp_path, project_db=True)
    review = _request(graph, nodes["goal_a"])
    graph.close()

    result = cli_runner.invoke(
        app,
        ["graph", "review", str(review.review_id), "accepted", "--project-root", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "interactive terminal" in result.output


def test_goal_review_cli_decides_from_confirmed_human_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, nodes = _hierarchy(tmp_path, project_db=True)
    review = _request(graph, nodes["goal_a"])
    graph.close()

    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
    graph.register_controller_master()

    import milknado.cli.graph as cli_graph

    monkeypatch.setattr(
        cli_graph,
        "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: True),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    monkeypatch.setattr(typer, "confirm", _confirm)
    monkeypatch.setattr(getpass, "getuser", lambda: "human-1")
    result = cli_runner.invoke(
        app,
        ["graph", "review", str(review.review_id), "accepted", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "human-1" in result.output
    graph = MikadoGraph(tmp_path / ".milknado" / "milknado.db")
    try:
        decided = graph.get_goal_review(review.review_id)
        assert decided is not None
        assert decided.decision is GoalReviewDecision.ACCEPTED
        assert decided.decided_by == "human-1"
    finally:
        graph.close()
