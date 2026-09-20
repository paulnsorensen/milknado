"""AC-6: selecting a node opens run/session/details/changes in the sidecar."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

import pytest
from playwright.sync_api import Page, expect

from milknado.adapters import ChangedFile
from milknado.app.run_source import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    NodeSnapshotRequest,
    RunActionAvailability,
)
from milknado.domains.common import MikadoEdge, MikadoNode, NodeKind, SessionContext, SessionView
from milknado.domains.common.session import SessionEvent
from milknado.domains.graph import (
    GraphSnapshot,
    NodeDetailResponse,
    NodeDetailSnapshot,
    NodeSessionSnapshot,
    RunRecord,
    SnapshotPage,
    SnapshotValue,
)
from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import (
    BROWSER_TOKEN,
    FIXTURE_NODE_DESCRIPTION,
    BrowserServer,
    BrowserSnapshotSource,
)

pytestmark = pytest.mark.browser

CHILD_DESCRIPTION = "Child task"
RUN_ID = "run-1"

_T = TypeVar("_T")


def _empty_page() -> SnapshotPage[_T]:
    return SnapshotPage(items=(), offset=0, limit=50, total=0, has_more=False)


def _run_record() -> RunRecord:
    return {
        "run_id": RUN_ID,
        "node_id": 2,
        "status": "running",
        "pid": None,
        "log_path": "",
        "started_at": "",
        "ended_at": None,
        "timed_out": False,
        "exit_code": None,
        "error": None,
        "timeout_seconds": None,
        "detail": None,
        "rebased": None,
    }


def _detail_response(request: NodeSnapshotRequest) -> NodeDetailResponse:
    session_page = SnapshotPage(
        items=(
            SessionEvent(kind="assistant", text=f"Transcript page {request.session_event_page}"),
        ),
        offset=0,
        limit=50,
        total=2,
        has_more=request.session_event_page == 0,
    )
    return NodeDetailResponse(
        node_id=request.node_id,
        request_generation=request.request_generation,
        detail=NodeDetailSnapshot(
            node=MikadoNode(
                id=request.node_id, description=CHILD_DESCRIPTION, kind=NodeKind.TASK, parent_id=1
            ),
            description=CHILD_DESCRIPTION,
            parent=None,
            children=_empty_page(),
            ancestors=_empty_page(),
            prerequisite_ids=_empty_page(),
            dependent_ids=_empty_page(),
            reverse_dependents=_empty_page(),
            owned_files=SnapshotPage(
                items=(f"file-page-{request.page}.py",),
                offset=0,
                limit=50,
                total=2,
                has_more=request.page == 0,
            ),
            runs=SnapshotPage(items=(_run_record(),), offset=0, limit=50, total=1, has_more=False),
            reviews=_empty_page(),
            sessions=SnapshotPage(
                items=(
                    NodeSessionSnapshot(
                        run_id=RUN_ID, session=None, state="loaded", event_history=session_page
                    ),
                ),
                offset=0,
                limit=50,
                total=1,
                has_more=False,
            ),
            receipts=_empty_page(),
            goal_claim=SnapshotValue(value=None, state="not_stored"),
            artifacts=_empty_page(),
        ),
    )


class _FakeGit:
    def changes(self, context: SessionContext) -> tuple[ChangedFile, ...]:
        del context
        return (ChangedFile(path="a.py", status="modified", added=1, removed=0, old_path=None),)

    def diff(self, context: SessionContext, path: str) -> str:
        del context
        return f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"


def _fixture_snapshot() -> ExecutionSnapshot:
    root = MikadoNode(id=1, description=FIXTURE_NODE_DESCRIPTION, kind=NodeKind.GOAL)
    child = MikadoNode(id=2, description=CHILD_DESCRIPTION, kind=NodeKind.TASK, parent_id=1)
    graph = GraphSnapshot(nodes=(root, child), edges=(MikadoEdge(1, 2),), root_ids=(1,))
    run = ActiveRunSnapshot(
        run_id=RUN_ID,
        node_id=2,
        description=CHILD_DESCRIPTION,
        status=ExecutionRunStatus.RUNNING,
        progress=None,
        stop_requested=False,
        actions=RunActionAvailability(),
        output=(),
        pending_guidance=None,
        elapsed_seconds=0.0,
        progress_pct=None,
        eta_seconds=None,
        attempt=None,
        max_attempts=None,
        stalled=False,
        session=SessionView(context=SessionContext(family="claude", cwd="/tmp/session")),
    )
    return ExecutionSnapshot(
        goal="Tracer fixture goal",
        active_runs=(run,),
        terminal_runs=(),
        completed=0,
        failed=0,
        stopped=0,
        available=1,
        event_lines=(),
        listener_errors=(),
        graph=graph,
        node=None,
    )


@pytest.fixture
def node_detail_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource(snapshot=_fixture_snapshot())
    source.node_snapshot = _detail_response  # type: ignore[method-assign]
    commands = WebCommands(git=_FakeGit())
    app = create_app(source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


def test_node_sidecar_shows_run_paging_and_changes(
    page: Page, node_detail_server: BrowserServer
) -> None:
    _ = page.goto(node_detail_server.login_url)
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True).click()

    expect(page.get_by_text(RUN_ID)).to_be_visible()
    expect(page.get_by_text("assistant: Transcript page 0")).to_be_visible()

    page.get_by_role("button", name="Next").click()
    expect(page.get_by_text("assistant: Transcript page 1")).to_be_visible()

    page.get_by_role("button", name="Previous").click()
    expect(page.get_by_text("assistant: Transcript page 0")).to_be_visible()

    page.get_by_role("button", name="Details").click()
    expect(page.get_by_text("file-page-0.py")).to_be_visible()

    page.get_by_role("button", name="Next").click()
    expect(page.get_by_text("file-page-1.py")).to_be_visible()

    page.get_by_role("button", name="Previous").click()
    expect(page.get_by_text("file-page-0.py")).to_be_visible()

    page.get_by_role("button", name="Changes").click()
    expect(page.get_by_text("a.py")).to_be_visible()

    page.get_by_text("a.py").click()
    expect(page.get_by_text("-old", exact=False)).to_be_visible()
