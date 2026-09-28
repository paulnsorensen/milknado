"""AC-9: observer capabilities disable owner-only controls with a visible reason."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from playwright.sync_api import Page, expect

from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import BROWSER_TOKEN, BrowserServer, BrowserSnapshotSource, open_app
from tests.browser.graph_db import GraphDbServer, graph_db_server

_ = graph_db_server

pytestmark = pytest.mark.browser

CASES: tuple[tuple[str, str], ...] = (("Force stop", "Force stop is unavailable."),)


@pytest.fixture
def observer_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource()
    app = create_app(source, WebCommands(), login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


@pytest.mark.parametrize("case", CASES, ids=[label for label, _ in CASES])
def test_observer_control_disabled_with_reason(
    page: Page, observer_server: BrowserServer, case: tuple[str, str]
) -> None:
    label, reason = case
    button = page.get_by_role("button", name=label, exact=True)
    open_app(page, observer_server.login_url, button)

    expect(button).to_be_disabled()
    expect(page.get_by_text(reason)).to_be_visible()


def test_observer_header_omits_stop_scheduling(page: Page, observer_server: BrowserServer) -> None:
    header_badge = page.locator('[data-region="header"]').get_by_text("Read-only", exact=True)
    open_app(page, observer_server.login_url, header_badge)

    expect(header_badge).to_be_visible()
    expect(page.get_by_role("button", name="Stop scheduling", exact=True)).not_to_be_visible()


def test_observer_selected_node_shows_unavailable_metrics_and_caption(
    page: Page, graph_db_server: GraphDbServer
) -> None:
    node = page.locator("button.mk-node", has_text="Edit target")
    open_app(page, graph_db_server.login_url, node)
    node.click()

    sidecar = page.locator('[data-region="sidecar"]')
    expect(sidecar.get_by_text("ETA", exact=True)).to_be_visible()
    expect(sidecar.get_by_text("Attempt", exact=True)).to_be_visible()
    expect(sidecar.get_by_text("guidance", exact=True)).to_be_visible()
    expect(sidecar.get_by_text("unavailable", exact=True)).to_have_count(3)
    expect(
        sidecar.locator('[data-region="sidecar-section"]').get_by_text("Read-only", exact=True)
    ).to_be_visible()
    expect(sidecar.get_by_label("Session guidance")).not_to_be_visible()
