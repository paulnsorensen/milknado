from collections.abc import Callable

from starlette.testclient import TestClient

from milknado.app.run_source import (
    ExecutionSnapshot,
    ExecutionSnapshotSource,
    NodeSnapshotRequest,
)
from milknado.domains.common import MikadoNode
from milknado.domains.graph import (
    NodeDetailResponse,
    NodeDetailSnapshot,
    SnapshotPage,
    SnapshotValue,
)
from milknado.web import LaunchToken, WebCommands, create_app


class FixtureSnapshotSource:
    _snapshot: ExecutionSnapshot
    snapshot_calls: int

    def __init__(self, goal: str = "fixture goal") -> None:
        self.snapshot_calls = 0
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
        self.snapshot_calls += 1
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
    test_client.cookies.set(login.cookie_name, login.value)  # pyright: ignore[reportUnknownMemberType]
    return test_client, login


def headers() -> dict[str, str]:
    return {"host": "127.0.0.1", "origin": "http://127.0.0.1"}


class RecordingCommands:
    def __init__(self) -> None:
        self.cancel_calls: list[str] = []
        self.force_stop_calls: list[str] = []
        self.stop_scheduling_calls: int = 0

    def cancel(self, run_id: str) -> dict[str, object]:
        self.cancel_calls.append(run_id)
        return {
            "run_id": run_id,
            "status": "cancelled",
            "terminal": True,
            "terminal_reason": "cancelled by request",
        }

    def force_stop(self, run_id: str) -> dict[str, object]:
        self.force_stop_calls.append(run_id)
        return {"run_id": run_id}

    def stop_scheduling(self) -> None:
        self.stop_scheduling_calls += 1


def recording_commands() -> tuple[WebCommands, RecordingCommands]:
    recording = RecordingCommands()
    return WebCommands(
        cancel=recording.cancel,
        force_stop=recording.force_stop,
        stop_scheduling=recording.stop_scheduling,
    ), recording


def client_with_source(
    commands: WebCommands | None = None,
) -> tuple[TestClient, LaunchToken, FixtureSnapshotSource]:
    login = LaunchToken("test-token")
    fixture = FixtureSnapshotSource()
    app = create_app(fixture, commands or WebCommands(), login)
    test_client = TestClient(app, base_url="http://127.0.0.1")
    test_client.cookies.set(login.cookie_name, login.value)  # pyright: ignore[reportUnknownMemberType]
    return test_client, login, fixture


def node_detail_response(node_id: int = 7) -> NodeDetailResponse:
    node = MikadoNode(id=node_id, description="fixture node")
    nodes = SnapshotPage((), 0, 10, 0, False)
    ids = SnapshotPage((), 0, 10, 0, False)
    detail = NodeDetailSnapshot(
        node=node,
        description=node.description,
        parent=None,
        children=nodes,
        ancestors=nodes,
        prerequisite_ids=ids,
        dependent_ids=ids,
        reverse_dependents=nodes,
        owned_files=SnapshotPage((), 0, 10, 0, False),
        runs=SnapshotPage((), 0, 10, 0, False),
        reviews=SnapshotPage((), 0, 10, 0, False),
        sessions=SnapshotPage((), 0, 10, 0, False),
        receipts=SnapshotPage((), 0, 10, 0, False),
        goal_claim=SnapshotValue(None, "not_stored"),
        artifacts=SnapshotPage((), 0, 10, 0, False),
    )
    return NodeDetailResponse(node_id, 2, detail)


def authenticated_client(
    snapshot_source: ExecutionSnapshotSource,
    commands: WebCommands | None = None,
    *,
    raise_server_exceptions: bool = True,
) -> TestClient:
    login = LaunchToken("test-token")
    result = TestClient(
        create_app(snapshot_source, commands or WebCommands(), login),
        base_url="http://127.0.0.1",
        raise_server_exceptions=raise_server_exceptions,
    )
    result.cookies.set(login.cookie_name, login.value)  # pyright: ignore[reportUnknownMemberType]
    return result
