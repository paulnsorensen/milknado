"""AC-5: a cleared login cookie sends the browser back to the login page."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.browser.conftest import FIXTURE_NODE_DESCRIPTION, BrowserServer

pytestmark = pytest.mark.browser


def test_cleared_cookie_returns_to_login_page(page: Page, browser_server: BrowserServer) -> None:
    _ = page.goto(browser_server.login_url)
    expect(page.get_by_text(FIXTURE_NODE_DESCRIPTION)).to_be_visible()

    page.context.clear_cookies()
    browser_server.restart(browser_server.app)

    expect(page).to_have_url(f"{browser_server.base_url}/")
    expect(page.get_by_label("Paste the launch URL")).to_be_visible()
