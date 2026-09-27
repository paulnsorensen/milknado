"""AC-6: selecting a node opens run/session/details/changes in the sidecar."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

import pytest
from playwright.sync_api import Locator, Page, expect

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
    open_app,
)

pytestmark = pytest.mark.browser

CHILD_DESCRIPTION = "Child task"
RUN_ID = "run-1"

_T = TypeVar("_T")


def _expect_transcript_line(page: Page, page_text: str) -> None:
    """Assert one assistant transcript line is visible for `page_text`.

    The console splits each line into a `.mk-line-time` actor column and a
    text column, so the row no longer reads as one `kind: text` string.
    """
    line = page.locator(".mk-line", has_text=page_text)
    expect(line).to_be_visible()
    expect(line.locator(".mk-line-time")).to_have_text("assistant")


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
    description = CHILD_DESCRIPTION if request.node_id == 2 else FIXTURE_NODE_DESCRIPTION
    return NodeDetailResponse(
        node_id=request.node_id,
        request_generation=request.request_generation,
        detail=NodeDetailSnapshot(
            node=MikadoNode(
                id=request.node_id, description=description, kind=NodeKind.TASK, parent_id=1
            ),
            description=description,
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


def _exercise_detail_tabs(page: Page) -> tuple[Locator, Locator]:
    """Assert tab semantics and return the Changes and Details controls."""
    tablist = page.get_by_role("tablist", name="Node detail")
    expect(tablist).to_be_visible()
    expect(tablist.get_by_role("tab")).to_have_count(3)
    for label, slug in (("Session", "session"), ("Changes", "changes"), ("Details", "details")):
        tab = tablist.get_by_role("tab", name=label)
        panel = page.locator(f"#node-detail-panel-{slug}")
        expect(tab).to_have_attribute("aria-controls", f"node-detail-panel-{slug}")
        expect(panel).to_have_attribute("role", "tabpanel")
        expect(panel).to_have_attribute("aria-labelledby", f"node-detail-tab-{slug}")

    session_tab = tablist.get_by_role("tab", name="Session")
    changes_tab = tablist.get_by_role("tab", name="Changes")
    details_tab = tablist.get_by_role("tab", name="Details")
    expect(session_tab).to_have_attribute("aria-selected", "true")
    expect(session_tab).to_have_attribute("tabindex", "0")

    session_tab.press("ArrowRight")
    expect(changes_tab).to_be_focused()
    expect(changes_tab).to_have_attribute("aria-selected", "true")
    changes_tab.press("ArrowRight")
    expect(details_tab).to_be_focused()
    expect(details_tab).to_have_attribute("aria-selected", "true")
    details_tab.press("ArrowLeft")
    expect(changes_tab).to_be_focused()
    changes_tab.press("ArrowLeft")
    expect(session_tab).to_be_focused()
    expect(page.locator(".mk-sidecar-title")).to_have_text(CHILD_DESCRIPTION)
    return changes_tab, details_tab


def test_node_sidecar_shows_run_paging_and_changes(
    page: Page, node_detail_server: BrowserServer
) -> None:
    open_app(
        page,
        node_detail_server.login_url,
        page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True),
    )

    page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True).click()

    expect(page.get_by_text(RUN_ID)).to_be_visible()
    _expect_transcript_line(page, "Transcript page 0")

    changes_tab, details_tab = _exercise_detail_tabs(page)

    page.get_by_role("button", name="Next").click()
    _expect_transcript_line(page, "Transcript page 1")

    page.get_by_role("button", name="Previous").click()
    _expect_transcript_line(page, "Transcript page 0")

    details_tab.click()
    expect(page.get_by_text("file-page-0.py")).to_be_visible()

    page.get_by_role("button", name="Next").click()
    expect(page.get_by_text("file-page-1.py")).to_be_visible()

    page.get_by_role("button", name="Previous").click()
    expect(page.get_by_text("file-page-0.py")).to_be_visible()

    changes_tab.click()
    expect(page.get_by_text("a.py")).to_be_visible()

    page.get_by_text("a.py").click()
    expect(page.get_by_text("-old", exact=False)).to_be_visible()
