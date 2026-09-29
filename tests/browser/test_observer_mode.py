"""AC-9: observer capabilities disable owner-only controls with a visible reason."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from playwright.sync_api import Page, expect

from milknado.app.run_source import ActiveRunSnapshot, ExecutionRunStatus, RunActionAvailability
from milknado.domains.graph import OwnerCapabilities
from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import (
    BROWSER_TOKEN,
    FIXTURE_NODE_DESCRIPTION,
    BrowserServer,
    BrowserSnapshotSource,
    build_fixture_snapshot,
    open_app,
    owner_web_commands,
)
from tests.browser.graph_db import GraphDbServer, graph_db_server

_ = graph_db_server

pytestmark = pytest.mark.browser

CASES: tuple[tuple[str, str], ...] = (("Force stop", "Force stop is unavailable."),)


def _active_run(run_id: str) -> ActiveRunSnapshot:
    return ActiveRunSnapshot(
        run_id=run_id,
        node_id=1,
        description="Running fixture",
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


@pytest.fixture
def watch_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    snapshot = replace(
        build_fixture_snapshot(),
        active_runs=(_active_run("run-1"),),
    )
    source = BrowserSnapshotSource(snapshot)
    commands = WebCommands(
        owner_capabilities=OwnerCapabilities(
            run_id="run-1",
            node_id=1,
            invocation_id="inv-1",
            owner_incarnation="owner-1",
            actions=(),
            permission_ids=(),
            published_at="now",
        )
    )
    app = create_app(source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def owner_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    snapshot = replace(
        build_fixture_snapshot(),
        active_runs=(_active_run("run-1"), _active_run("run-2")),
    )
    source = BrowserSnapshotSource(snapshot)

    def owner_capabilities(run_id: str | None) -> OwnerCapabilities | None:
        if run_id is None:
            return None
        return OwnerCapabilities(
            run_id=run_id,
            node_id=1,
            invocation_id="inv-1",
            owner_incarnation="owner-1",
            actions=(),
            permission_ids=(),
            published_at="now",
        )

    commands, _ = owner_web_commands()
    commands = replace(commands, owner_capabilities=owner_capabilities)
    app = create_app(source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


@pytest.mark.parametrize("case", CASES, ids=[label for label, _ in CASES])
def test_observer_control_disabled_with_reason(
    page: Page, observer_server: BrowserServer, case: tuple[str, str]
) -> None:
    label, reason = case
    node_button = page.get_by_role(
        "button", name=f"pending {FIXTURE_NODE_DESCRIPTION}", exact=True
    )
    open_app(page, observer_server.login_url, node_button)
    node_button.click()

    button = page.get_by_role("button", name=label, exact=True)
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


def test_owner_header_shows_run_mode_with_two_active_runs(
    page: Page, owner_server: BrowserServer
) -> None:
    open_app(page, owner_server.login_url, page.get_by_text("Run active"))

    expect(page.get_by_text("Run active")).to_be_visible()
    expect(page.get_by_text("Run", exact=True)).to_be_visible()
    expect(page.get_by_text("Watch", exact=True)).not_to_be_visible()
    expect(page.get_by_role("button", name="Stop scheduling", exact=True)).to_be_visible()

    node = page.locator("button.mk-node", has_text="Tracer fixture node")
    open_app(page, owner_server.login_url, node)
    node.click()

    sidecar = page.locator('[data-region="sidecar"]')
    # With two active runs the per-run owner capability stays unresolved (node 110),
    # so the session input is not offered; the host-owner mode still hides the
    # observer-only "unavailable" rows.
    expect(sidecar.get_by_text("unavailable", exact=True)).to_have_count(0)


def test_watch_with_one_live_owner_shows_observer_surface(
    page: Page, watch_server: BrowserServer
) -> None:
    node = page.locator("button.mk-node", has_text="Tracer fixture node")
    open_app(page, watch_server.login_url, node)
    node.click()

    sidecar = page.locator('[data-region="sidecar"]')
    header_badge = page.locator('[data-region="header"]').get_by_text("Read-only", exact=True)
    expect(header_badge).to_be_visible()
    expect(page.get_by_role("button", name="Stop scheduling", exact=True)).not_to_be_visible()
    expect(sidecar.get_by_text("unavailable", exact=True)).to_have_count(3)
    expect(
        sidecar.locator('[data-region="sidecar-section"]').get_by_text("Read-only", exact=True)
    ).to_be_visible()
    expect(sidecar.get_by_label("Session guidance")).not_to_be_visible()
