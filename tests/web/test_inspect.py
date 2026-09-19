from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import pytest
from httpx import Response as HttpxResponse
from starlette.testclient import TestClient

from milknado.adapters import ChangedFile
from milknado.app.run_source import (
    ExecutionRunStatus,
    ExecutionSnapshot,
    NodeSnapshotRequest,
    TerminalRunSnapshot,
)
from milknado.domains.common import GitOperationError, SessionContext, SessionView
from milknado.domains.graph import NodeDetailResponse
from milknado.web import WebCommands
from tests.web.support import authenticated_client, headers, node_detail_response


@dataclass
class InspectionSource:
    detail: NodeDetailResponse
    run: TerminalRunSnapshot
    last_request: NodeSnapshotRequest | None = None

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
        self.last_request = request
        assert request.node_id == self.detail.node_id
        return self.detail


class LookupErrorSource(InspectionSource):
    last_request: NodeSnapshotRequest | None

    def node_snapshot(  # pyright: ignore[reportImplicitOverride]
        self, request: NodeSnapshotRequest
    ) -> NodeDetailResponse:
        self.last_request = request
        raise LookupError("snapshot source failed")


class InspectionGit:
    def changes(self, context: SessionContext) -> tuple[ChangedFile, ...]:
        assert context == SessionContext(family="test", cwd="/tmp")
        return (ChangedFile("README.md", "M", 2, 1),)

    def diff(self, context: SessionContext, path: str) -> str:
        assert context == SessionContext(family="test", cwd="/tmp")
        assert path == "README.md"
        return "diff -- README.md"


class FailingGit:
    def changes(self, context: SessionContext) -> tuple[ChangedFile, ...]:
        raise GitOperationError("status", f"failed for {context.cwd}")

    def diff(self, context: SessionContext, path: str) -> str:
        raise GitOperationError("diff", f"failed for {context.cwd}:{path}")


class InvalidPathGit(InspectionGit):
    def diff(  # pyright: ignore[reportImplicitOverride]
        self, context: SessionContext, path: str
    ) -> str:
        _ = context
        raise ValueError(f"invalid diff path: {path}")


def _detail(node_id: int = 7) -> NodeDetailResponse:
    return node_detail_response(node_id)


def _run() -> TerminalRunSnapshot:
    return TerminalRunSnapshot(
        run_id="run-1",
        node_id=7,
        description="fixture",
        status=ExecutionRunStatus.COMPLETED,
        output=(),
        pending_guidance=None,
        duration_seconds=0,
        session=SessionView(context=SessionContext(family="test", cwd="/tmp")),
    )


def request(client: TestClient, path: str) -> HttpxResponse:
    return cast(
        HttpxResponse,
        client.get(  # pyright: ignore[reportUnknownMemberType]
            path, headers=headers()
        ),
    )


def client(
    *,
    detail: NodeDetailResponse | None = None,
    run: TerminalRunSnapshot | None = None,
    git: InspectionGit | FailingGit | InvalidPathGit | None = None,
    include_git: bool = True,
) -> TestClient:
    source = InspectionSource(detail=detail or _detail(), run=run or _run())
    commands = WebCommands(git=git if include_git else None)
    return authenticated_client(source, commands)


def client_from_source(
    source: InspectionSource, *, raise_server_exceptions: bool = True
) -> TestClient:
    return authenticated_client(source, raise_server_exceptions=raise_server_exceptions)


def test_inspection_routes_return_fixture_data() -> None:
    source = InspectionSource(detail=_detail(), run=_run())
    response = client_from_source(source).get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/nodes/7?page=1&limit=10&session_event_page=2", headers=headers()
    )
    assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
    assert source.last_request == NodeSnapshotRequest(
        node_id=7,
        request_generation=0,
        page=1,
        limit=10,
        session_event_page=2,
    )

    default_source = InspectionSource(detail=_detail(), run=_run())
    response = request(client_from_source(default_source), "/api/nodes/7")
    assert response.status_code == 200
    assert default_source.last_request == NodeSnapshotRequest(node_id=7, request_generation=0)

    encoded = cast(dict[str, object], response.json())
    detail = cast(dict[str, object], encoded["detail"])
    node = cast(dict[str, object], detail["node"])
    assert encoded["node_id"] == 7
    assert encoded["request_generation"] == 2
    assert node["id"] == 7
    assert detail["description"] == "fixture node"

    response = client(git=InspectionGit()).get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/runs/run-1/changes", headers=headers()
    )
    assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
    assert response.json() == [  # pyright: ignore[reportUnknownMemberType]
        {"path": "README.md", "status": "M", "added": 2, "removed": 1, "old_path": None}
    ]

    response = client(git=InspectionGit()).get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/runs/run-1/diff?path=README.md", headers=headers()
    )
    assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
    assert response.text == "diff -- README.md"  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("limit=-1", "limit must be non-negative"),
        ("limit=0", "limit must be between 1 and 100"),
        ("limit=101", "limit must be between 1 and 100"),
        ("page=invalid", "page must be an integer"),
        ("session_event_page=-1", "session_event_page must be non-negative"),
    ],
)
def test_node_detail_rejects_invalid_query_bounds(query: str, reason: str) -> None:
    source = InspectionSource(detail=_detail(), run=_run())
    response = request(client_from_source(source), f"/api/nodes/7?{query}")
    assert response.status_code == 400
    assert response.json() == {"error": reason}
    assert source.last_request is None


def test_node_detail_returns_not_found_for_unknown_node() -> None:
    response = request(
        client(detail=NodeDetailResponse(node_id=99, request_generation=0, detail=None)),
        "/api/nodes/99",
    )
    assert response.status_code == 404
    assert response.json() == {"error": "Node 99 was not found."}


def test_run_inspection_rejects_unknown_run() -> None:
    response = request(client(), "/api/runs/missing/changes")
    assert response.status_code == 404
    assert response.json() == {"error": "Run missing was not found."}


def test_run_inspection_rejects_missing_session_worktree() -> None:
    response = request(
        client(run=replace(_run(), session=SessionView()), git=InspectionGit()),
        "/api/runs/run-1/changes",
    )
    assert response.status_code == 409
    assert response.json() == {"error": "Run has no session worktree."}


def test_run_inspection_rejects_missing_git_capability() -> None:
    response = request(client(include_git=False), "/api/runs/run-1/changes")
    assert response.status_code == 409
    assert response.json() == {"error": "Git inspection is unavailable."}


@pytest.mark.parametrize(
    ("endpoint", "reason"),
    [
        ("changes", "git status failed: failed for /tmp"),
        ("diff?path=README.md", "git diff failed: failed for /tmp:README.md"),
    ],
)
def test_run_inspection_reports_git_failures(endpoint: str, reason: str) -> None:
    response = request(client(git=FailingGit()), f"/api/runs/run-1/{endpoint}")
    assert response.status_code == 409
    assert response.json() == {"error": reason}


def test_diff_rejects_invalid_path() -> None:
    response = request(client(git=InvalidPathGit()), "/api/runs/run-1/diff?path=../secret")
    assert response.status_code == 400
    assert response.json() == {"error": "invalid diff path: ../secret"}


def test_node_detail_does_not_mislabel_snapshot_source_failure() -> None:
    response = request(
        client_from_source(
            LookupErrorSource(detail=_detail(), run=_run()),
            raise_server_exceptions=False,
        ),
        "/api/nodes/7",
    )
    assert response.status_code == 500
