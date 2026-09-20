"""AC-15: at 400px, the narrow list and detail views hold no horizontal scroll."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from playwright.sync_api import Page, ViewportSize, expect

from milknado.app.run_source import NodeSnapshotRequest
from milknado.domains.common import MikadoNode, NodeKind
from milknado.domains.graph import (
    NodeDetailResponse,
    NodeDetailSnapshot,
    SnapshotPage,
    SnapshotValue,
)
from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import BROWSER_TOKEN, BrowserServer, BrowserSnapshotSource

pytestmark = pytest.mark.browser

NARROW_VIEWPORT: ViewportSize = {"width": 400, "height": 800}
NODE_DESCRIPTION = "Tracer fixture node"
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
    _ = page.goto(narrow_server.login_url)
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("treeitem", name=NODE_DESCRIPTION)).to_be_visible()
    expect(page.get_by_label("Jump to node")).to_be_visible()
    assert _has_no_horizontal_scroll(page)


def test_narrow_detail_view_is_full_width_with_44px_controls_and_no_horizontal_scroll(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    _ = page.goto(narrow_server.login_url)
    page.wait_for_load_state("networkidle")

    page.get_by_role("treeitem", name=NODE_DESCRIPTION).click()
    page.get_by_role("button", name="Open node").click()

    back_button = page.get_by_role("button", name="Back to list")
    expect(back_button).to_be_visible()
    box = back_button.bounding_box()
    assert box is not None
    assert box["width"] >= MIN_TOUCH_TARGET
    assert box["height"] >= MIN_TOUCH_TARGET

    assert _has_no_horizontal_scroll(page)
