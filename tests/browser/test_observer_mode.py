"""AC-9: observer capabilities disable owner-only controls with a visible reason."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from playwright.sync_api import Page, expect

from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import BROWSER_TOKEN, BrowserServer, BrowserSnapshotSource

pytestmark = pytest.mark.browser

CASES: tuple[tuple[str, str], ...] = (
    ("Force stop", "Force stop is unavailable."),
    ("Stop scheduling", "Stop scheduling is unavailable."),
)


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
    _ = page.goto(observer_server.login_url)
    page.wait_for_load_state("networkidle")

    button = page.get_by_role("button", name=label, exact=True)
    expect(button).to_be_visible()
    expect(button).to_be_disabled()
    expect(page.get_by_text(reason)).to_be_visible()
