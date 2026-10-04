"""AC-10: a 409's domain reason shows a toast; listener errors show a banner."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest
from playwright.sync_api import Page, expect

from milknado.app.watch import WatchSnapshotSource
from milknado.cli._helpers import ensure_db
from milknado.domains.graph import OwnerCapabilities
from milknado.project import load_project_config
from milknado.web import LaunchToken, PolledSnapshotSource, WebCommands, create_app
from milknado.web.commands import GraphEditCommands
from tests.browser.conftest import (
    BROWSER_TOKEN,
    FIXTURE_NODE_DESCRIPTION,
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
        host_owner=True,
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
    node_button = page.get_by_role(
        "button", name=f"pending {FIXTURE_NODE_DESCRIPTION}", exact=True
    )
    open_app(
        page,
        unavailable_cancel_server.login_url,
        node_button,
    )
    node_button.click()

    page.get_by_role("button", name="Cancel run", exact=True).click()
    page.get_by_role("alertdialog").get_by_role("button", name="Cancel run", exact=True).click()

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


@pytest.fixture(params=("missing", "unreadable"), ids=("no-db", "unreadable-db"))
def recovered_project_server(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[tuple[BrowserServer, Path, str]]:
    project_root = tmp_path / "project"
    config = load_project_config(project_root)
    state = cast(Literal["missing", "unreadable"], request.param)
    if state == "unreadable":
        config.db_path.parent.mkdir(parents=True)
        _ = config.db_path.write_bytes(b"not a sqlite database")
    graph = ensure_db(config)
    source = PolledSnapshotSource(WatchSnapshotSource(project_root, config.db_path), interval=0.05)
    source.start()
    login = LaunchToken(BROWSER_TOKEN)
    commands = WebCommands(
        graph_edits=GraphEditCommands(
            graph=graph,
            flavor_registry=getattr(config, "flavor_registry", frozenset()),
            project_root=project_root,
        )
    )
    server = BrowserServer(app=create_app(source, commands, login), login=login)
    server.start()
    try:
        yield server, config.db_path, state
    finally:
        server.stop()
        source.close()
        graph.close()


def test_empty_and_unreadable_projects_render_empty_dashboard(
    page: Page, recovered_project_server: tuple[BrowserServer, Path, str]
) -> None:
    server, db_path, state = recovered_project_server
    open_app(page, server.login_url, page.locator(".mk-shell"))

    expect(page.locator(".mk-graph-node")).to_have_count(0)
    expect(page.get_by_text("No goal reviews are pending.")).to_be_visible()
    assert db_path.is_file()
    if state == "unreadable":
        quarantined = tuple(db_path.parent.glob(f"{db_path.name}.corrupt-*"))
        assert quarantined


def test_listener_error_shows_banner(page: Page, listener_error_server: BrowserServer) -> None:
    open_app(
        page, listener_error_server.login_url, page.get_by_text("Live update listener failed.")
    )
