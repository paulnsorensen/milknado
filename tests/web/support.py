# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from types import SimpleNamespace
from typing import cast

from starlette.testclient import TestClient

from milknado.app.run_source import ExecutionSnapshotSource
from milknado.web import LaunchToken, WebCommands, create_app


def source(goal: str = "fixture goal") -> ExecutionSnapshotSource:
    snapshot = SimpleNamespace(
        goal=goal,
        active_runs=(),
        terminal_runs=(),
        completed=0,
        failed=0,
        stopped=0,
        available=1,
        event_lines=(),
        graph=None,
        node=None,
    )
    return cast(ExecutionSnapshotSource, cast(object, SimpleNamespace(snapshot=lambda: snapshot)))


def client(commands: WebCommands | None = None) -> tuple[TestClient, LaunchToken]:
    login = LaunchToken("test-token")
    app = create_app(source(), commands or WebCommands(), login)
    test_client = TestClient(app, base_url="http://127.0.0.1")
    test_client.cookies.set(login.cookie_name, login.value)
    return test_client, login


def headers() -> dict[str, str]:
    return {"host": "127.0.0.1", "origin": "http://127.0.0.1"}
