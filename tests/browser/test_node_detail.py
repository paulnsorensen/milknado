"""AC-6: selecting a node opens run/session/details/changes in the sidecar."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import TypeVar, cast

import pytest
from playwright.sync_api import Locator, Page, expect

from milknado.adapters import ChangedFile
from milknado.app.run_source import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    NodeSnapshotRequest,
    RunActionAvailability,
    TerminalRunSnapshot,
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
LONG_DESCRIPTION = (
    "A long sidecar description that must clamp before the detail controls. " * 12
).strip()
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


def _run_record(status: str = "running", error: str | None = None) -> RunRecord:
    return {
        "run_id": RUN_ID,
        "node_id": 2,
        "status": status,
        "pid": None,
        "log_path": "",
        "started_at": "",
        "ended_at": None,
        "timed_out": False,
        "exit_code": None,
        "error": error,
        "timeout_seconds": None,
        "detail": None,
        "rebased": None,
    }


def _detail_response(
    request: NodeSnapshotRequest,
    *,
    description: str = CHILD_DESCRIPTION,
    run_record: RunRecord | None = None,
) -> NodeDetailResponse:
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
            runs=SnapshotPage(
                items=(run_record or _run_record(),), offset=0, limit=50, total=1, has_more=False
            ),
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


def _long_detail_response(request: NodeSnapshotRequest) -> NodeDetailResponse:
    return _detail_response(request, description=LONG_DESCRIPTION)


def _failed_detail_response(request: NodeSnapshotRequest) -> NodeDetailResponse:
    return _detail_response(
        request,
        run_record=_run_record(status="failed", error="worker session gone"),
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


def _no_worktree_snapshot() -> ExecutionSnapshot:
    return replace(
        _fixture_snapshot(),
        active_runs=(),
        terminal_runs=(
            TerminalRunSnapshot(
                run_id=RUN_ID,
                node_id=2,
                description=CHILD_DESCRIPTION,
                status=ExecutionRunStatus.FAILED,
                output=(),
                pending_guidance=None,
                duration_seconds=0.0,
            ),
        ),
    )


@pytest.fixture
def long_description_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource(snapshot=_fixture_snapshot())
    source.node_snapshot = _long_detail_response  # type: ignore[method-assign]
    app = create_app(source, WebCommands(git=_FakeGit()), login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def no_worktree_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource(snapshot=_no_worktree_snapshot())
    source.node_snapshot = _failed_detail_response  # type: ignore[method-assign]
    app = create_app(source, WebCommands(git=_FakeGit()), login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def missing_changes_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    snapshot = replace(_fixture_snapshot(), active_runs=(), terminal_runs=())
    source = BrowserSnapshotSource(snapshot=snapshot)
    source.node_snapshot = _detail_response  # type: ignore[method-assign]
    app = create_app(source, WebCommands(git=_FakeGit()), login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


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


LONG_PARENT = (
    "A parent description that must remain concise in the breadcrumb. "
    "Additional context stays outside the path item."
)
LONG_PARENT_SUMMARY = "A parent description that must remain concise in the breadcrumb."


@pytest.fixture
def long_path_server() -> Iterator[BrowserServer]:
    nodes = [
        MikadoNode(id=1, description=LONG_PARENT, kind=NodeKind.GOAL),
        MikadoNode(id=3, description="Ancestor three", kind=NodeKind.TASK, parent_id=1),
        MikadoNode(id=4, description="Ancestor four", kind=NodeKind.TASK, parent_id=3),
        MikadoNode(id=2, description=CHILD_DESCRIPTION, kind=NodeKind.TASK, parent_id=4),
    ]
    graph = GraphSnapshot(
        nodes=tuple(nodes),
        edges=tuple(
            MikadoEdge(parent.id, child.id)
            for parent, child in zip(nodes, nodes[1:], strict=False)
        ),
        root_ids=(1,),
    )
    source = BrowserSnapshotSource(snapshot=replace(_fixture_snapshot(), graph=graph))
    source.node_snapshot = _detail_response  # type: ignore[method-assign]
    login = LaunchToken(BROWSER_TOKEN)
    app = create_app(source, WebCommands(git=_FakeGit()), login)
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


def test_missing_changes_endpoint_is_an_empty_state_without_page_error(
    page: Page, missing_changes_server: BrowserServer
) -> None:
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    open_app(
        page,
        missing_changes_server.login_url,
        page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True),
    )

    page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True).click()
    page.get_by_role("tab", name="Changes").click()

    expect(page.get_by_text("No changes")).to_be_visible()
    assert page_errors == []


def test_long_description_shows_expand_control_only_when_clamped(
    page: Page, long_description_server: BrowserServer
) -> None:
    open_app(
        page,
        long_description_server.login_url,
        page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True),
    )

    page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True).click()

    expect(page.get_by_role("button", name="Expand description", exact=True)).to_be_visible()
    page.get_by_role("button", name="Expand description", exact=True).click()
    expect(page.get_by_role("button", name="Collapse description", exact=True)).to_be_visible()


def test_empty_sidecar_has_no_controls_before_selection(
    page: Page, node_detail_server: BrowserServer
) -> None:
    open_app(
        page,
        node_detail_server.login_url,
        page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True),
    )

    sidecar = page.get_by_role("complementary", name="Detail")
    expect(sidecar).to_contain_text("Select a node to inspect its details.")
    expect(sidecar.locator("button")).to_have_count(0)
    expect(sidecar.locator("textarea")).to_have_count(0)


def test_failed_run_without_worktree_uses_real_no_changes_state(
    page: Page, no_worktree_server: BrowserServer
) -> None:
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    open_app(
        page,
        no_worktree_server.login_url,
        page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True),
    )

    page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True).click()

    expect(page.get_by_role("complementary", name="Detail").get_by_role("alert")).to_contain_text(
        "worker session gone"
    )
    page.get_by_role("tab", name="Changes").click()
    expect(page.get_by_text("No changes")).to_be_visible()
    assert page_errors == []


def test_ancestor_path_caps_at_four_items_with_single_line_ellipsis(
    page: Page, long_path_server: BrowserServer
) -> None:
    open_app(
        page,
        long_path_server.login_url,
        page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True),
    )

    page.get_by_role("button", name=f"pending {CHILD_DESCRIPTION}", exact=True).click()
    expect(page.locator(".mk-path-gap")).to_have_count(0)
    path_item = page.get_by_role("button", name=LONG_PARENT_SUMMARY, exact=True)
    expect(path_item).to_have_text(LONG_PARENT_SUMMARY)
    expect(path_item).to_have_attribute("aria-label", LONG_PARENT_SUMMARY)

    style = cast(
        dict[str, str],
        page.locator(".mk-path-item").first.evaluate(
            """(element) => ({
                maxWidth: getComputedStyle(element).maxWidth,
                overflow: getComputedStyle(element).overflow,
                textOverflow: getComputedStyle(element).textOverflow,
                whiteSpace: getComputedStyle(element).whiteSpace
            })"""
        ),
    )
    assert style == {
        "maxWidth": "160px",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
        "whiteSpace": "nowrap",
    }
