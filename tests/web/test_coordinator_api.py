# pyright: basic
import asyncio
from pathlib import Path
from typing import cast

import msgspec
import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from milknado.domains.common import CONTROLLER_MASTER_ENV
from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import (
    DecideGoalReview,
    RequestGoalReview,
    StartGoal,
)
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.graph import GoalReviewDecision, MikadoGraph
from milknado.web import LaunchToken, WebCommands, create_app
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
