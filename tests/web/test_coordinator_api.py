# pyright: basic
import asyncio
import sqlite3
from pathlib import Path
from typing import cast

import msgspec
import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from milknado.cli.web import _host_dependencies
from milknado.domains.batching import BatchPlan
from milknado.domains.common import CONTROLLER_MASTER_ENV, MilknadoConfig, SessionInput
from milknado.domains.coordinator import CoordinatorControl, ProviderBinding
from milknado.domains.coordinator.control_models import (
    DecideGoalReview,
    PlanGoal,
    Recover,
    RequestGoalReview,
    RuntimeAction,
    StartGoal,
)
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.coordinator.persistence import bind_provider_session, link_entity
from milknado.domains.coordinator.recovery import (
    ProviderIdentity,
    ProviderTurn,
    record_provider_turn,
)
from milknado.domains.graph import GoalReviewDecision, MikadoGraph
from milknado.domains.planning import PlanChangeManifest, Planner, PlanProposal, PlanResult
from milknado.web import LaunchToken, WebCommands, create_app
from milknado.web.commands import GraphEditCommands
from milknado.web.routes.coordinator import _events
from tests.web.support import FixtureSnapshotSource, headers


class CoordinatorStub:
    def __init__(self) -> None:
        self.commands: list[tuple[str, object]] = []

    def list_coordinator_sessions(self) -> list[dict[str, str]]:
        return [{"id": "session-1", "description": "Deliver"}]

    def read_coordinator_snapshot(self, session_id: str, cursor: int) -> dict[str, object]:
        if session_id != "session-1":
            raise KeyError(session_id)
        return {"session": {"id": session_id}, "cursor": cursor, "events": []}

    def send_coordinator_command(self, session_id: str, command: object) -> dict[str, object]:
        self.commands.append((session_id, command))
        return {"command_id": "cmd-1", "status": "unavailable"}


def test_coordinator_routes_require_login_and_validate_cursor() -> None:
    login = LaunchToken("test-token")
    stub = CoordinatorStub()
    app = create_app(FixtureSnapshotSource(), WebCommands(coordinator=stub), login)
    client = TestClient(app, base_url="http://127.0.0.1")
    assert client.get("/api/coordinators/session-1/snapshot").status_code == 401
    client.cookies.set(login.cookie_name, login.value)
    assert client.get("/api/coordinators/session-1/snapshot?cursor=3").json()["cursor"] == 3
    assert client.get("/api/coordinators/session-1/snapshot?cursor=-1").status_code == 400
    assert client.get("/api/coordinators/foreign/snapshot").status_code == 404
    assert client.get("/api/coordinators").json() == [
        {"id": "session-1", "description": "Deliver"}
    ]


def test_plan_approval_api_requires_login_and_applies_once(tmp_path: Path) -> None:
    class PlannerStub:
        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            assert (goal, project_root) == ("Deliver", tmp_path)
            assert target_goal_id > 0
            manifest = PlanChangeManifest("milknado.plan.v2", goal, goal, None, (), ())
            return PlanProposal(manifest, tmp_path / "context.md")

        def prepare_proposal(self, proposal: PlanProposal, project_root: Path) -> BatchPlan:
            _ = (proposal, project_root)
            return BatchPlan((), (), "OPTIMAL")

        def apply_proposal(
            self,
            proposal: PlanProposal,
            *,
            target_goal_id: int,
            prepared_plan: BatchPlan,
        ) -> PlanResult:
            _ = prepared_plan
            _ = graph.add_node("Applied task", target_goal_id)
            return PlanResult(True, 0, proposal.context_path, nodes_created=1)

    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(planner=cast(Planner, cast(object, PlannerStub())))
    )
    login = LaunchToken("test-token")
    client = TestClient(
        create_app(FixtureSnapshotSource(), WebCommands(coordinator=control), login),
        base_url="http://127.0.0.1",
    )
    start_body = {
        "kind": "start_goal",
        "command_id": "start",
        "description": "Deliver",
        "provider": "codex",
    }
    assert (
        client.post("/api/coordinators/commands", json=start_body, headers=headers()).status_code
        == 401
    )
    client.cookies.set(login.cookie_name, login.value)
    session = client.post("/api/coordinators/commands", json=start_body, headers=headers()).json()[
        "result"
    ]
    route = f"/api/coordinators/{session['id']}/commands"
    proposed = client.post(
        route, json={"kind": "plan_goal", "command_id": "plan-1"}, headers=headers()
    )
    assert proposed.status_code == 200
    assert proposed.json()["result"]["status"] == "pending"
    assert graph.get_children(session["goal_id"]) == []
    decision = {
        "kind": "decide_plan_proposal",
        "command_id": "approve",
        "proposal_id": "plan-1",
        "decision": "accepted",
    }
    assert (
        client.post(route, json=decision, headers=headers()).json()["result"]["status"]
        == "applied"
    )
    assert len(graph.get_children(session["goal_id"])) == 1
    decision["command_id"] = "approve-again"
    assert (
        client.post(route, json=decision, headers=headers()).json()["result"]["status"]
        == "applied"
    )
    assert len(graph.get_children(session["goal_id"])) == 1
    graph.close()


def test_coordinator_stream_rejects_invalid_cursor_and_missing_session() -> None:
    login = LaunchToken("test-token")
    app = create_app(FixtureSnapshotSource(), WebCommands(coordinator=CoordinatorStub()), login)
    client = TestClient(app, base_url="http://127.0.0.1")
    client.cookies.set(login.cookie_name, login.value)
    stream = "/api/coordinators/session-1/stream"
    invalid = client.get(f"{stream}?cursor=-1")
    assert invalid.status_code == 400
    assert invalid.json() == {"error": "cursor must be a non-negative integer"}
    missing = client.get("/api/coordinators/foreign/stream")
    assert missing.status_code == 404
    assert missing.json() == {"error": "Coordinator session does not exist."}


def test_coordinator_routes_report_unavailable_port() -> None:
    login = LaunchToken("test-token")
    app = create_app(FixtureSnapshotSource(), WebCommands(), login)
    client = TestClient(app, base_url="http://127.0.0.1")
    client.cookies.set(login.cookie_name, login.value)
    assert client.get("/api/coordinators").status_code == 409
    session = "/api/coordinators/session-1"
    responses = (
        client.get(f"{session}/snapshot"),
        client.post(
            f"{session}/commands", json={"kind": "recover", "command_id": "cmd"}, headers=headers()
        ),
        client.get(f"{session}/stream"),
    )
    for response in responses:
        assert response.status_code == 409
        assert response.json() == {"error": "Coordinator is unavailable."}


def test_start_goal_rejects_session_command_route() -> None:
    login = LaunchToken("test-token")
    app = create_app(FixtureSnapshotSource(), WebCommands(coordinator=CoordinatorStub()), login)
    client = TestClient(app, base_url="http://127.0.0.1")
    client.cookies.set(login.cookie_name, login.value)
    payload = {
        "kind": "start_goal",
        "command_id": "start",
        "description": "Deliver",
        "provider": "codex",
    }
    response = client.post("/api/coordinators/session-1/commands", json=payload, headers=headers())
    assert response.status_code == 400
    assert response.json() == {"error": "start_goal uses /api/coordinators/commands"}


def test_coordinator_command_route_validates_and_delegates() -> None:
    login = LaunchToken("test-token")
    stub = CoordinatorStub()
    app = create_app(FixtureSnapshotSource(), WebCommands(coordinator=stub), login)
    client = TestClient(app, base_url="http://127.0.0.1")
    client.cookies.set(login.cookie_name, login.value)
    payload = {"kind": "recover", "command_id": "cmd-1"}
    response = client.post("/api/coordinators/session-1/commands", json=payload, headers=headers())
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert len(stub.commands) == 1
    assert (
        client.post(
            "/api/coordinators/session-1/commands", json={"kind": "recover"}, headers=headers()
        ).status_code
        == 400
    )


def test_review_route_requires_reviewer_identity() -> None:
    login = LaunchToken("test-token")
    stub = CoordinatorStub()
    app = create_app(FixtureSnapshotSource(), WebCommands(coordinator=stub), login)
    client = TestClient(app, base_url="http://127.0.0.1")
    client.cookies.set(login.cookie_name, login.value)
    payload = {
        "kind": "request_goal_review",
        "command_id": "review",
        "goal_revision": "rev-1",
        "evidence": "evidence",
        "proposed_change": "change",
    }
    url = "/api/coordinators/session-1/commands"
    assert client.post(url, json=payload, headers=headers()).status_code == 400
    payload["reviewer"] = "human"
    assert client.post(url, json=payload, headers=headers()).status_code == 200
    assert isinstance(stub.commands[-1][1], RequestGoalReview)
    assert stub.commands[-1][1].reviewer == "human"


def test_live_stream_projects_review_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "review-secret")
    graph = MikadoGraph(tmp_path / "graph.db")
    graph.register_controller_master()
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(review_decision=graph.decide_goal_review)
    )
    start = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], start.result)["id"])
    review = control.send_coordinator_command(
        session_id, RequestGoalReview("request", "rev-1", "evidence", "change", "human")
    )
    review_id = cast(int, cast(dict[str, object], review.result)["review_id"])
    cursor = control.read_coordinator_snapshot(session_id, 0).cursor
    initial = control.read_coordinator_snapshot(session_id, cursor)

    class Connected:
        async def is_disconnected(self) -> bool:
            return False

    async def receive() -> dict[str, object]:
        events = _events(cast(Request, Connected()), control, initial, (session_id, cursor))
        pending = asyncio.ensure_future(anext(events))
        await asyncio.sleep(0)
        control.send_coordinator_command(
            session_id,
            DecideGoalReview("decide", review_id, GoalReviewDecision.ACCEPTED, "human"),
        )
        event = await asyncio.wait_for(pending, 2)
        await events.aclose()
        return cast(dict[str, object], msgspec.json.decode(event["data"].encode()))

    snapshot = asyncio.run(receive())
    assert cast(int, snapshot["cursor"]) > cursor
    assert cast(list[dict[str, object]], snapshot["events"])[0]["status"] == "accepted"
    assert cast(list[dict[str, object]], snapshot["reviews"])[0]["decision"] == "accepted"
    graph.close()


def test_production_host_connects_planner_and_deferred_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PlannerStub:
        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            assert (goal, project_root) == ("Deliver", tmp_path)
            assert target_goal_id > 0
            manifest = PlanChangeManifest("milknado.plan.v2", goal, goal, None, (), ())
            return PlanProposal(manifest, tmp_path / "context.md")

    planner = PlannerStub()
    monkeypatch.setattr("milknado.app.plan.build_planner", lambda *_: planner)
    graph = MikadoGraph(tmp_path / "graph.db")
    config = MilknadoConfig(project_root=tmp_path, db_path=graph.db_path)
    dependencies = _host_dependencies(graph, config, tmp_path, (None, None))
    control = cast(CoordinatorControl, dependencies.coordinator)
    start = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], start.result)["id"])
    assert control.send_coordinator_command(session_id, PlanGoal("plan")).status == "accepted"
    identity = ProviderIdentity("codex", "thread-1")
    with sqlite3.connect(graph.db_path) as conn:
        link_entity(conn, session_id, "provider_session", identity.session_id)
        bind_provider_session(
            conn,
            session_id,
            ProviderBinding("coordinator", session_id, "codex", identity.session_id),
        )
        record_provider_turn(conn, session_id, ProviderTurn(identity, "turn-1", "submitted"))
    recovered = control.send_coordinator_command(session_id, Recover("recover"))
    assert recovered.status == "accepted"
    result = cast(dict[str, object], recovered.result)
    assert cast(list[dict[str, object]], result["receipts"])[0]["outcome"] == "unavailable"
    assert cast(list[dict[str, object]], result["unknown_turns"])[0]["turn_id"] == "turn-1"
    assert (
        control.send_coordinator_command(
            session_id, RuntimeAction("action", "provider-1", SessionInput(action="interrupt"))
        ).status
        == "unavailable"
    )
    graph.close()


def test_legacy_http_decision_advances_coordinator_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "review-secret")
    graph = MikadoGraph(tmp_path / "graph.db")
    graph.register_controller_master()
    control = CoordinatorControl(
        graph, tmp_path, CoordinatorServices(review_decision=graph.decide_goal_review)
    )
    start = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], start.result)["id"])
    pending = control.send_coordinator_command(
        session_id, RequestGoalReview("request", "rev", "evidence", "change", "human")
    )
    review_id = cast(int, cast(dict[str, object], pending.result)["review_id"])
    cursor = control.read_coordinator_snapshot(session_id, 0).cursor
    commands = WebCommands(
        coordinator=control,
        review_decision=control.decide_goal_review,
        graph_edits=GraphEditCommands(graph, frozenset(), tmp_path),
    )
    client = TestClient(
        create_app(FixtureSnapshotSource(), commands, LaunchToken("test-token")),
        base_url="http://127.0.0.1",
    )
    client.cookies.set("milknado_login", "test-token")
    response = client.post(
        f"/api/reviews/{review_id}/decision",
        json={"decision": "accepted", "decided_by": "human"},
        headers=headers(),
    )
    assert response.status_code == 200
    assert response.json()["decision"] == "accepted"
    after = control.read_coordinator_snapshot(session_id, cursor)
    assert after.cursor > cursor
    assert [(event.kind, event.status) for event in after.events] == [("approval", "accepted")]
    graph.close()


def test_fresh_database_coordinator_routes_return_404(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    client = TestClient(
        create_app(
            FixtureSnapshotSource(), WebCommands(coordinator=control), LaunchToken("test-token")
        ),
        base_url="http://127.0.0.1",
    )
    client.cookies.set("milknado_login", "test-token")
    root = "/api/coordinators/missing"
    responses = (
        client.get(f"{root}/snapshot"),
        client.post(
            f"{root}/commands",
            json={"kind": "recover", "command_id": "recover"},
            headers=headers(),
        ),
        client.get(f"{root}/stream"),
    )
    assert [response.status_code for response in responses] == [404, 404, 404]
    graph.close()
