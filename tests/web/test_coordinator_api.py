# pyright: basic
import asyncio
from pathlib import Path
from typing import cast

import msgspec
import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from milknado.cli.web import _host_dependencies
from milknado.domains.common import CONTROLLER_MASTER_ENV, MilknadoConfig, SessionInput
from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import (
    DecideGoalReview,
    PlanGoal,
    Recover,
    RequestGoalReview,
    RuntimeAction,
    StartGoal,
)
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.graph import GoalReviewDecision, MikadoGraph
from milknado.domains.planning import PlanResult
from milknado.web import LaunchToken, WebCommands, create_app
from milknado.web.commands import GraphEditCommands
from milknado.web.routes.coordinator import _events
from tests.web.support import FixtureSnapshotSource, headers


class CoordinatorStub:
    def __init__(self) -> None:
        self.commands: list[tuple[str, object]] = []

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


def test_production_host_connects_planner_without_provider_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PlannerStub:
        def launch(
            self, goal: str, project_root: Path, *, target_goal_id: int | None = None
        ) -> PlanResult:
            assert (goal, project_root) == ("Deliver", tmp_path)
            assert target_goal_id is not None
            return PlanResult(True, 0, tmp_path / "context.md", nodes_created=0)

    planner = PlannerStub()
    monkeypatch.setattr("milknado.app.plan.build_planner", lambda *_: planner)
    graph = MikadoGraph(tmp_path / "graph.db")
    config = MilknadoConfig(project_root=tmp_path, db_path=graph.db_path)
    dependencies = _host_dependencies(graph, config, tmp_path, (None, None))
    control = cast(CoordinatorControl, dependencies.coordinator)
    start = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], start.result)["id"])
    assert control.send_coordinator_command(session_id, PlanGoal("plan")).status == "accepted"
    assert control.send_coordinator_command(session_id, Recover("recover")).status == "unavailable"
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
