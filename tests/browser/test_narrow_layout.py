"""AC-15: at 400px, the narrow list and detail views hold no horizontal scroll."""

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from playwright.sync_api import Page, ViewportSize, expect

from milknado.app.run_source import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    NodeSnapshotRequest,
    RunActionAvailability,
)
from milknado.domains.common import MikadoNode, NodeKind
from milknado.domains.graph import (
    NodeDetailResponse,
    NodeDetailSnapshot,
    SnapshotPage,
    SnapshotValue,
)
from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import (
    BROWSER_TOKEN,
    BrowserServer,
    BrowserSnapshotSource,
    open_app,
)

pytestmark = pytest.mark.browser

NARROW_VIEWPORT: ViewportSize = {"width": 400, "height": 800}
ROSTER_VIEWPORT: ViewportSize = {"width": 800, "height": 800}
NODE_DESCRIPTION = "Tracer fixture node"

LONG_AGENT_DESCRIPTION = (
    "Deterministic fixture only. Do not inspect the repository, edit files, "
    "or report success with a fake receipt."
)
MIN_TOUCH_TARGET = 44


def _empty_page() -> SnapshotPage[Any]:  # pyright: ignore[reportExplicitAny]
    return SnapshotPage(items=(), offset=0, limit=50, total=0, has_more=False)


def _detail_response(request: NodeSnapshotRequest) -> NodeDetailResponse:
    return NodeDetailResponse(
        node_id=request.node_id,
        request_generation=request.request_generation,
        detail=NodeDetailSnapshot(
            node=MikadoNode(id=request.node_id, description=NODE_DESCRIPTION, kind=NodeKind.GOAL),
            description=NODE_DESCRIPTION,
            parent=None,
            children=_empty_page(),
            ancestors=_empty_page(),
            prerequisite_ids=_empty_page(),
            dependent_ids=_empty_page(),
            reverse_dependents=_empty_page(),
            owned_files=_empty_page(),
            runs=_empty_page(),
            reviews=_empty_page(),
            sessions=_empty_page(),
            receipts=_empty_page(),
            goal_claim=SnapshotValue(value=None, state="not_stored"),
            artifacts=_empty_page(),
        ),
    )


@pytest.fixture
def narrow_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource()
    source.node_snapshot = _detail_response  # type: ignore[method-assign]
    commands = WebCommands()
    app = create_app(source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


def _has_no_horizontal_scroll(page: Page) -> bool:
    overflow = page.evaluate(  # pyright: ignore[reportAny]
        "() => document.scrollingElement.scrollWidth <= document.scrollingElement.clientWidth"
    )
    return bool(overflow)  # pyright: ignore[reportAny]


def test_narrow_list_view_shows_outline_with_no_horizontal_scroll(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, narrow_server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    expect(page.get_by_label("Jump to node")).to_be_visible()
    assert _has_no_horizontal_scroll(page)


def test_narrow_detail_view_is_full_width_with_44px_controls_and_no_horizontal_scroll(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, narrow_server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    page.get_by_role("treeitem", name=NODE_DESCRIPTION).click()
    page.get_by_role("button", name="Open node").click()

    back_button = page.get_by_role("button", name="Back to list")
    expect(back_button).to_be_visible()
    box = back_button.bounding_box()
    assert box is not None
    assert box["width"] >= MIN_TOUCH_TARGET
    assert box["height"] >= MIN_TOUCH_TARGET

    assert _has_no_horizontal_scroll(page)


def test_narrow_detail_tabs_use_aria_roles_and_roving_focus(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, narrow_server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    page.get_by_role("treeitem", name=NODE_DESCRIPTION).click()
    page.get_by_role("button", name="Open node").click()

    tablist = page.get_by_role("tablist", name="Node detail")
    expect(tablist).to_be_visible()
    expect(tablist.get_by_role("tab")).to_have_count(3)

    tabs = {
        label: tablist.get_by_role("tab", name=label)
        for label in ("Session", "Changes", "Details")
    }
    for label, slug in (("Session", "session"), ("Changes", "changes"), ("Details", "details")):
        tab = tabs[label]
        panel = page.locator(f"#node-detail-panel-{slug}")
        expect(tab).to_have_attribute("aria-controls", f"node-detail-panel-{slug}")
        expect(panel).to_have_attribute("role", "tabpanel")
        expect(panel).to_have_attribute("aria-labelledby", f"node-detail-tab-{slug}")

    session = tabs["Session"]
    changes = tabs["Changes"]
    details = tabs["Details"]
    expect(session).to_have_attribute("aria-selected", "true")
    expect(session).to_have_attribute("tabindex", "0")

    session.press("ArrowRight")
    expect(changes).to_be_focused()
    expect(changes).to_have_attribute("aria-selected", "true")
    changes.press("ArrowRight")
    expect(details).to_be_focused()
    details.press("ArrowRight")
    expect(session).to_be_focused()


def test_long_agent_description_is_clamped_in_the_roster(
    page: Page, browser_server: BrowserServer, browser_source: BrowserSnapshotSource
) -> None:
    page.set_viewport_size(ROSTER_VIEWPORT)
    browser_source.publish(
        replace(
            browser_source.snapshot(),
            active_runs=(
                ActiveRunSnapshot(
                    run_id="run-long-description",
                    node_id=1,
                    description=LONG_AGENT_DESCRIPTION,
                    status=ExecutionRunStatus.RUNNING,
                    progress=None,
                    stop_requested=False,
                    actions=RunActionAvailability(),
                    output=(),
                    pending_guidance=None,
                    elapsed_seconds=0,
                    progress_pct=None,
                    eta_seconds=None,
                    attempt=None,
                    max_attempts=None,
                    stalled=False,
                ),
            ),
            available=0,
        )
    )
    agent_name = page.locator(".mk-agent-roster .mk-agent-name b").first
    open_app(page, browser_server.login_url, agent_name)

    expect(agent_name).to_be_visible()
    expect(agent_name).to_have_text(LONG_AGENT_DESCRIPTION)
    expect(agent_name).to_have_css("display", "block")
    expect(agent_name).to_have_css("overflow", "hidden")
    expect(agent_name).to_have_css("text-overflow", "ellipsis")
    expect(agent_name).to_have_css("white-space", "nowrap")

    agent_row = agent_name.locator("..").locator("..")
    agent_box = agent_name.bounding_box()
    row_box = agent_row.bounding_box()
    assert agent_box is not None
    assert row_box is not None
    assert agent_box["height"] <= 16
    assert row_box["height"] <= 32
