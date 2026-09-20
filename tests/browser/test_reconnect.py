"""AC-4: the reconnecting status appears while the stream is down and clears."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.browser.conftest import (
    FIXTURE_NODE_DESCRIPTION,
    BrowserServer,
    BrowserSnapshotSource,
    build_fixture_snapshot,
)

pytestmark = pytest.mark.browser


def test_stream_drop_shows_status_then_clears_on_reconnect(
    page: Page, browser_server: BrowserServer, browser_source: BrowserSnapshotSource
) -> None:
    reconnecting = page.get_by_text("Connection lost. The browser tries again.")
    _ = page.goto(browser_server.login_url)
    expect(page.get_by_text(FIXTURE_NODE_DESCRIPTION)).to_be_visible()
    expect(reconnecting).to_have_count(0)

    browser_server.stop()

    expect(reconnecting).to_be_visible(timeout=15000)

    with page.expect_response(
        lambda response: response.url.endswith("/api/stream") and response.status == 200
    ):
        browser_server.start()
    browser_source.publish(build_fixture_snapshot())

    expect(reconnecting).to_have_count(0, timeout=15000)
    expect(page.get_by_text(FIXTURE_NODE_DESCRIPTION)).to_be_visible()
