"""AC-8: confirming a run-control action mints one command; dismiss mints none."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest
from playwright.sync_api import Page

from milknado.domains.graph import OwnerCapabilities
from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import (
    BROWSER_TOKEN,
    BrowserServer,
    BrowserSnapshotSource,
    RecordingCommands,
    owner_web_commands,
)

pytestmark = pytest.mark.browser

RUN_ID = "run-1"

CASES: tuple[tuple[str, Callable[[RecordingCommands], int]], ...] = (
    ("Cancel run", lambda recorder: len(recorder.cancel_calls)),
    ("Force stop", lambda recorder: len(recorder.force_stop_calls)),
    ("Stop scheduling", lambda recorder: recorder.stop_scheduling_calls),
)


@pytest.fixture
def confirm_server() -> Iterator[tuple[BrowserServer, RecordingCommands]]:
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


def _wait_for_count(get_count: Callable[[], int], expected: int) -> None:
    deadline = time.monotonic() + 5.0
    while get_count() != expected:
        if time.monotonic() > deadline:
            raise TimeoutError(f"expected {expected} calls, got {get_count()}")
        time.sleep(0.02)


@pytest.mark.parametrize("case", CASES, ids=[label for label, _ in CASES])
def test_confirm_records_one_command(
    page: Page,
    confirm_server: tuple[BrowserServer, RecordingCommands],
    case: tuple[str, Callable[[RecordingCommands], int]],
) -> None:
    trigger_label, call_count = case
    server, recorder = confirm_server
    _ = page.goto(server.login_url)
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name=trigger_label, exact=True).click()
    page.get_by_role("button", name="Confirm").click()

    _wait_for_count(lambda: call_count(recorder), 1)


@pytest.mark.parametrize("case", CASES, ids=[label for label, _ in CASES])
def test_dismiss_records_zero_commands(
    page: Page,
    confirm_server: tuple[BrowserServer, RecordingCommands],
    case: tuple[str, Callable[[RecordingCommands], int]],
) -> None:
    trigger_label, call_count = case
    server, recorder = confirm_server
    _ = page.goto(server.login_url)
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name=trigger_label, exact=True).click()
    page.get_by_role("button", name="Dismiss").click()
    page.wait_for_timeout(200)

    assert call_count(recorder) == 0
