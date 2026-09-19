"""AC-1: the committed build renders the graph over a real browser + server."""

from __future__ import annotations

from typing import cast

import pytest
from playwright.sync_api import Page

from tests.browser.conftest import FIXTURE_NODE_DESCRIPTION, BrowserServer

pytestmark = pytest.mark.browser


def _goto_logged_in(page: Page, server: BrowserServer) -> None:
    _ = page.goto(server.login_url)
    page.wait_for_load_state("networkidle")


def test_graph_renders_fixture_node_with_shared_react(
    page: Page, browser_server: BrowserServer
) -> None:
    console_messages: list[str] = []
    page.on("console", lambda message: console_messages.append(message.text))

    _goto_logged_in(page, browser_server)

    assert page.get_by_text(FIXTURE_NODE_DESCRIPTION).first.is_visible()

    shares_react = cast(bool, page.evaluate("window.Milknado.React === window.React"))
    assert shares_react is True

    invalid_hook_calls = [text for text in console_messages if "Invalid hook call" in text]
    assert invalid_hook_calls == []
