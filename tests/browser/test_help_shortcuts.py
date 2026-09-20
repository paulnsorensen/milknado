"""AC-16: Help lists shortcuts, keyboard input drives them, typing is safe."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from playwright.sync_api import Page, expect

from milknado.app.run_source import ExecutionSnapshot
from milknado.domains.common import MikadoNode, NodeKind
from milknado.domains.graph import GraphSnapshot, OwnerCapabilities
from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import (
    BROWSER_TOKEN,
    FIXTURE_NODE_DESCRIPTION,
    BrowserServer,
    BrowserSnapshotSource,
    RecordingCommands,
    open_app,
    owner_web_commands,
    wait_until,
)

pytestmark = pytest.mark.browser

SECOND_NODE_DESCRIPTION = "Second fixture node"
RUN_ID = "run-1"


def _two_root_snapshot() -> ExecutionSnapshot:
    first = MikadoNode(id=1, description=FIXTURE_NODE_DESCRIPTION, kind=NodeKind.GOAL)
    second = MikadoNode(id=2, description=SECOND_NODE_DESCRIPTION, kind=NodeKind.GOAL)
    graph = GraphSnapshot(nodes=(first, second), edges=(), root_ids=(1, 2))
    return ExecutionSnapshot(
        goal="Tracer fixture goal",
        active_runs=(),
        terminal_runs=(),
        completed=0,
        failed=0,
        stopped=0,
        available=1,
        event_lines=(),
        listener_errors=(),
        graph=graph,
        node=None,
    )


@pytest.fixture
def two_root_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource(snapshot=_two_root_snapshot())
    commands, _ = owner_web_commands()
    app = create_app(source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def steering_server() -> Iterator[tuple[BrowserServer, RecordingCommands]]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource()
    commands, recorder = owner_web_commands()
    commands = WebCommands(
        cancel=commands.cancel,
        force_stop=commands.force_stop,
        stop_scheduling=commands.stop_scheduling,
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
    yield server, recorder
    server.stop()


def test_question_mark_opens_help_with_all_columns(
    page: Page, two_root_server: BrowserServer
) -> None:
    open_app(
        page,
        two_root_server.login_url,
        page.get_by_role("button", name=f"pending {FIXTURE_NODE_DESCRIPTION}", exact=True),
    )

    page.keyboard.press("?")

    dialog = page.get_by_role("dialog", name="Keyboard shortcuts")
    dialog.wait_for(state="visible")
    for column in ("Graph", "Runs", "Steering"):
        assert dialog.get_by_role("region", name=column).is_visible()


def test_arrow_key_moves_graph_selection(page: Page, two_root_server: BrowserServer) -> None:
    first = page.get_by_role("button", name=f"pending {FIXTURE_NODE_DESCRIPTION}", exact=True)
    second = page.get_by_role("button", name=f"pending {SECOND_NODE_DESCRIPTION}", exact=True)
    open_app(page, two_root_server.login_url, first)
    first.click()

    page.keyboard.press("ArrowDown")
    page.wait_for_timeout(50)

    expect(second).to_have_attribute("aria-selected", "true")
    expect(first).to_have_attribute("aria-selected", "false")


def test_steering_key_confirms_and_runs_command(
    page: Page, steering_server: tuple[BrowserServer, RecordingCommands]
) -> None:
    server, recorder = steering_server
    open_app(
        page,
        server.login_url,
        page.get_by_role("button", name=f"pending {FIXTURE_NODE_DESCRIPTION}", exact=True),
    )

    page.keyboard.press("x")
    page.get_by_role("button", name="Confirm").click()

    wait_until(lambda: len(recorder.cancel_calls) == 1)


def test_shortcut_key_is_ignored_while_typing(page: Page, two_root_server: BrowserServer) -> None:
    open_app(
        page,
        two_root_server.login_url,
        page.get_by_role("button", name=f"pending {FIXTURE_NODE_DESCRIPTION}", exact=True),
    )

    page.get_by_role("textbox", name="Jump to node").click()
    page.keyboard.press("?")
    page.wait_for_timeout(200)

    assert page.get_by_role("dialog", name="Keyboard shortcuts").count() == 0
