"""AC-10: a 409's domain reason shows a toast; listener errors show a banner."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from playwright.sync_api import Page, expect

from milknado.domains.graph import OwnerCapabilities
from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import (
    BROWSER_TOKEN,
    BrowserServer,
    BrowserSnapshotSource,
    build_fixture_snapshot,
    open_app,
)

pytestmark = pytest.mark.browser

RUN_ID = "run-1"


@pytest.fixture
def unavailable_cancel_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource()
    commands = WebCommands(
        owner_capabilities=OwnerCapabilities(
            run_id=RUN_ID,
            node_id=1,
            invocation_id="invocation-1",
            owner_incarnation="1",
            actions=(),
            permission_ids=(),
            published_at="",
        ),
    )
    app = create_app(source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


def test_409_domain_reason_shows_toast(
    page: Page, unavailable_cancel_server: BrowserServer
) -> None:
    open_app(
        page,
        unavailable_cancel_server.login_url,
        page.get_by_role("button", name="Cancel run", exact=True),
    )

    page.get_by_role("button", name="Cancel run", exact=True).click()
    page.get_by_role("button", name="Confirm").click()

    expect(page.get_by_text("Cancel is unavailable.")).to_be_visible()


@pytest.fixture
def listener_error_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    snapshot = replace(build_fixture_snapshot(), listener_errors=("Live update listener failed.",))
    source = BrowserSnapshotSource(snapshot=snapshot)
    app = create_app(source, WebCommands(), login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


def test_listener_error_shows_banner(page: Page, listener_error_server: BrowserServer) -> None:
    open_app(
        page, listener_error_server.login_url, page.get_by_text("Live update listener failed.")
    )
