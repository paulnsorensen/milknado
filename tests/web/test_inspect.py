from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from starlette.testclient import TestClient

from milknado.adapters import ChangedFile
from milknado.app.run_source import (
    ExecutionRunStatus,
    ExecutionSnapshot,
    NodeSnapshotRequest,
    TerminalRunSnapshot,
)
from milknado.domains.common import SessionContext, SessionView
from milknado.domains.graph import NodeDetailResponse, NodeDetailSnapshot
from milknado.web import LaunchToken, WebCommands, create_app


@dataclass
class InspectionSource:
    detail: NodeDetailResponse
    run: TerminalRunSnapshot

    def snapshot(self) -> ExecutionSnapshot:
        return ExecutionSnapshot(
            goal="fixture goal",
            active_runs=(),
            terminal_runs=(self.run,),
            completed=1,
            failed=0,
            stopped=0,
            available=0,
            event_lines=(),
        )

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        del listener
        return lambda: None

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        assert request.node_id == 7
        return self.detail


class InspectionGit:
    def changes(self, run_id: str) -> tuple[ChangedFile, ...]:
        assert run_id == "run-1"
        return (ChangedFile("README.md", "M", 2, 1),)

    def diff(self, run_id: str, path: str) -> str:
        assert run_id == "run-1"
        assert path == "README.md"
        return "diff -- README.md"


def client() -> TestClient:
    source = InspectionSource(
        NodeDetailResponse(
            node_id=7,
            request_generation=2,
            detail=cast(NodeDetailSnapshot, cast(object, {})),
        ),
        TerminalRunSnapshot(
            run_id="run-1",
            node_id=7,
            description="fixture",
            status=ExecutionRunStatus.COMPLETED,
            output=(),
            pending_guidance=None,
            duration_seconds=0,
            session=SessionView(context=SessionContext(family="test", cwd="/tmp")),
        ),
    )
    login = LaunchToken("test-token")
    app = create_app(source, WebCommands(git=InspectionGit()), login)
    result = TestClient(app, base_url="http://127.0.0.1")
    result.cookies.set(  # pyright: ignore[reportUnknownMemberType]
        login.cookie_name, login.value
    )
    return result


def headers() -> dict[str, str]:
    return {"host": "127.0.0.1", "origin": "http://127.0.0.1"}


def test_inspection_routes_return_fixture_data() -> None:
    response = client().get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/nodes/7?page=1&limit=10&session_event_page=2", headers=headers()
    )
    assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
    assert response.json() == {  # pyright: ignore[reportUnknownMemberType]
        "node_id": 7,
        "request_generation": 2,
        "detail": {},
    }

    response = client().get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/runs/run-1/changes", headers=headers()
    )
    assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
    assert response.json() == [  # pyright: ignore[reportUnknownMemberType]
        {"path": "README.md", "status": "M", "added": 2, "removed": 1, "old_path": None}
    ]

    response = client().get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/runs/run-1/diff?path=README.md", headers=headers()
    )
    assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
    assert response.text == "diff -- README.md"  # pyright: ignore[reportUnknownMemberType]
