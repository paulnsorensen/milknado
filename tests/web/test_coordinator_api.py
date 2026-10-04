# pyright: basic
from starlette.testclient import TestClient

from milknado.web import LaunchToken, WebCommands, create_app
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
