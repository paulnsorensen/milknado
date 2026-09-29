"""AC-8: confirming a run-control action mints one command; dismiss mints none."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from playwright.sync_api import Page

from milknado.domains.graph import OwnerCapabilities
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

RUN_ID = "run-1"

CASES: tuple[tuple[str, Callable[[RecordingCommands], int]], ...] = (
    ("Cancel run", lambda recorder: len(recorder.cancel_calls)),
    ("Force stop", lambda recorder: len(recorder.force_stop_calls)),
    ("Stop scheduling", lambda recorder: recorder.stop_scheduling_calls),
)

CONFIRM_LABELS = {
    "Cancel run": ("Cancel run", "Keep the run"),
    "Force stop": ("Force stop", "Keep the run"),
    "Stop scheduling": ("Stop runs", "Keep running"),
}
CONFIRM_BODIES = {
    "Cancel run": "The action runs once. It cannot be undone from here.",
    "Force stop": (
        "The run stops now. It does not wait for the current turn. "
        "Changes that are not committed stay in the worktree."
    ),
    "Stop scheduling": (
        "Milknado dispatches no more nodes. Each active run stops after its current turn. "
        "Done work stays in the graph."
    ),
}


@pytest.fixture
def confirm_server() -> Iterator[tuple[BrowserServer, RecordingCommands]]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource()
    commands, recorder = owner_web_commands()
    commands = WebCommands(
        cancel=commands.cancel,
        force_stop=commands.force_stop,
        stop_scheduling=commands.stop_scheduling,
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
    yield server, recorder
    server.stop()


@pytest.mark.parametrize("case", CASES, ids=[label for label, _ in CASES])
def test_confirm_records_one_command(
    page: Page,
    confirm_server: tuple[BrowserServer, RecordingCommands],
    case: tuple[str, Callable[[RecordingCommands], int]],
) -> None:
    trigger_label, call_count = case
    confirm_label, _ = CONFIRM_LABELS[trigger_label]
    server, recorder = confirm_server
    node_button = page.get_by_role(
        "button", name=f"pending {FIXTURE_NODE_DESCRIPTION}", exact=True
    )
    open_app(page, server.login_url, node_button)
    node_button.click()
    controls = page.get_by_role("region", name="Run controls")
    trigger = (
        page.get_by_role("button", name=trigger_label, exact=True)
        if trigger_label == "Stop scheduling"
        else controls.get_by_role("button", name=trigger_label, exact=True)
    )
    trigger.click()
    dialog = page.get_by_role("alertdialog")
    assert dialog.get_by_text(CONFIRM_BODIES[trigger_label], exact=True).is_visible()
    dialog.get_by_role("button", name=confirm_label, exact=True).click()

    wait_until(lambda: call_count(recorder) == 1)


@pytest.mark.parametrize("case", CASES, ids=[label for label, _ in CASES])
def test_dismiss_records_zero_commands(
    page: Page,
    confirm_server: tuple[BrowserServer, RecordingCommands],
    case: tuple[str, Callable[[RecordingCommands], int]],
) -> None:
    trigger_label, call_count = case
    _, dismiss_label = CONFIRM_LABELS[trigger_label]
    server, recorder = confirm_server
    node_button = page.get_by_role(
        "button", name=f"pending {FIXTURE_NODE_DESCRIPTION}", exact=True
    )
    open_app(page, server.login_url, node_button)
    node_button.click()
    controls = page.get_by_role("region", name="Run controls")
    trigger = (
        page.get_by_role("button", name=trigger_label, exact=True)
        if trigger_label == "Stop scheduling"
        else controls.get_by_role("button", name=trigger_label, exact=True)
    )
    trigger.click()
    page.get_by_role("alertdialog").get_by_role("button", name=dismiss_label, exact=True).click()
    page.wait_for_timeout(200)

    assert call_count(recorder) == 0
