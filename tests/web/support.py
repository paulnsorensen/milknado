# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from collections.abc import Callable

from starlette.testclient import TestClient

from milknado.app.run_source import (
    ExecutionSnapshot,
    ExecutionSnapshotSource,
    NodeSnapshotRequest,
)
from milknado.domains.graph import NodeDetailResponse
from milknado.web import LaunchToken, WebCommands, create_app


class FixtureSnapshotSource:
    _snapshot: ExecutionSnapshot

    def __init__(self, goal: str = "fixture goal") -> None:
        self._snapshot = ExecutionSnapshot(
            goal=goal,
            active_runs=(),
            terminal_runs=(),
            completed=0,
            failed=0,
            stopped=0,
            available=1,
            event_lines=(),
            listener_errors=("fixture listener error",),
            graph=None,
            node=None,
        )
        self._listeners: list[Callable[[ExecutionSnapshot], None]] = []

    def snapshot(self) -> ExecutionSnapshot:
        return self._snapshot

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def publish(self, snapshot: ExecutionSnapshot) -> None:
        self._snapshot = snapshot
        for listener in tuple(self._listeners):
            listener(snapshot)

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        raise NotImplementedError(request)


def source(goal: str = "fixture goal") -> ExecutionSnapshotSource:
    return FixtureSnapshotSource(goal)


def client(commands: WebCommands | None = None) -> tuple[TestClient, LaunchToken]:
    login = LaunchToken("test-token")
    app = create_app(source(), commands or WebCommands(), login)
    test_client = TestClient(app, base_url="http://127.0.0.1")
    test_client.cookies.set(login.cookie_name, login.value)
    return test_client, login


def headers() -> dict[str, str]:
    return {"host": "127.0.0.1", "origin": "http://127.0.0.1"}
