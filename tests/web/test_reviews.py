# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false
from pathlib import Path

from starlette.testclient import TestClient

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.graph import GoalReviewRequest, MikadoGraph
from milknado.web import LaunchToken, WebCommands, create_app
from milknado.web.commands import GraphEditCommands
from tests.web.support import FixtureSnapshotSource, headers


def test_reviews_list_and_decision(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
        graph.request_goal_review(
            GoalReviewRequest(goal.id, "rev", "evidence", "change", reviewer="reviewer")
        )
        graph.register_controller_master()
        commands = WebCommands(
            graph_edits=GraphEditCommands(graph, frozenset({"implement"}), tmp_path),
            review_decision=lambda request, *, decided_by: graph.decide_goal_review(
                request, decided_by=decided_by
            ),
        )
        app = create_app(FixtureSnapshotSource(), commands, LaunchToken("test-token"))
        client = TestClient(app, base_url="http://127.0.0.1")
        client.cookies.set("milknado_login", "test-token")
        listed = client.get("/api/reviews", headers=headers())
        assert listed.status_code == 200
        assert listed.json()[0]["review_id"] == 1
        decided = client.post(
            "/api/reviews/1/decision",
            json={
                "decision": "accepted",
                "decided_by": "web",
                "decided_at": "2026-09-19T15:00:00+00:00",
            },
            headers=headers(),
        )
        assert decided.status_code == 200
        decision = decided.json()
        assert decision["decision"] == "accepted"
        assert decision["decided_by"] == "web"
        assert decision["decided_at"] == "2026-09-19T15:00:00+00:00"
        assert client.get("/api/reviews", headers=headers()).json() == []
    finally:
        graph.close()


def test_review_route_rejects_unknown_fields_without_mutation(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
        graph.request_goal_review(
            GoalReviewRequest(goal.id, "rev", "evidence", "change", reviewer="reviewer")
        )
        graph.register_controller_master()
        commands = WebCommands(
            graph_edits=GraphEditCommands(graph, frozenset({"implement"}), tmp_path),
            review_decision=lambda request, *, decided_by: graph.decide_goal_review(
                request, decided_by=decided_by
            ),
        )
        app = create_app(FixtureSnapshotSource(), commands, LaunchToken("test-token"))
        client = TestClient(app, base_url="http://127.0.0.1")
        client.cookies.set("milknado_login", "test-token")
        rejected = client.post(
            "/api/reviews/1/decision",
            json={"decision": "accepted", "extra": True},
            headers=headers(),
        )
        assert rejected.status_code == 400
        record = graph.get_goal_review(1)
        assert record is not None
        assert record.decision.value == "pending"
    finally:
        graph.close()


def test_review_route_errors_and_non_goal_nodes(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        _ = graph.add_node("task")
        commands = WebCommands(
            graph_edits=GraphEditCommands(graph, frozenset({"implement"}), tmp_path),
            review_decision=lambda request, *, decided_by: (_ for _ in ()).throw(
                ValueError("invalid review")
            ),
        )
        app = create_app(FixtureSnapshotSource(), commands, LaunchToken("test-token"))
        client = TestClient(app, base_url="http://127.0.0.1")
        client.cookies.set("milknado_login", "test-token")
        assert client.get("/api/reviews", headers=headers()).json() == []
        assert (
            client.post("/api/reviews/1/decision", content=b"{", headers=headers()).status_code
            == 400
        )
        assert (
            client.post(
                "/api/reviews/1/decision", json={"decision": "accepted"}, headers=headers()
            ).status_code
            == 409
        )
    finally:
        graph.close()
