"""AC-7: send input, steer, interrupt, approve and deny each mint one command."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from playwright.sync_api import Page

from milknado.domains.common import SessionInput
from milknado.domains.graph import OwnerCapabilities
from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import (
    BROWSER_TOKEN,
    BrowserServer,
    BrowserSnapshotSource,
    RecordingCommands,
    open_app,
    owner_web_commands,
    wait_until,
)

pytestmark = pytest.mark.browser

RUN_ID = "run-1"
PERMISSION_ID = "perm-1"


@pytest.fixture
def session_input_server() -> Iterator[tuple[BrowserServer, RecordingCommands]]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource()
    commands, recorder = owner_web_commands()
    commands = WebCommands(
        session_input=commands.session_input,
        owner_capabilities=OwnerCapabilities(
            run_id=RUN_ID,
            node_id=1,
            invocation_id="invocation-1",
            owner_incarnation="1",
            actions=("steer", "follow_up", "interrupt", "approve", "deny"),
            permission_ids=(PERMISSION_ID,),
            published_at="",
        ),
    )
    app = create_app(source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server, recorder
    server.stop()


def _wait_for_call_count(recorder: RecordingCommands, count: int) -> tuple[str, SessionInput]:
    wait_until(lambda: len(recorder.session_input_calls) >= count)
    assert len(recorder.session_input_calls) == count
    return recorder.session_input_calls[-1]


def test_session_input_buttons_each_mint_one_fresh_command(
    page: Page, session_input_server: tuple[BrowserServer, RecordingCommands]
) -> None:
    server, recorder = session_input_server
    open_app(page, server.login_url, page.get_by_label("Session guidance"))

    seen_command_ids: list[str] = []

    def _send_and_check(action: str, button_name: str, *, fill: bool) -> None:
        if fill:
            page.get_by_label("Session guidance").fill(f"guidance for {action}")
        page.get_by_role("button", name=button_name).click()
        run_id, request = _wait_for_call_count(recorder, len(seen_command_ids) + 1)
        assert run_id == RUN_ID
        assert request.action == action
        assert request.command_id != ""
        assert request.command_id not in seen_command_ids
        seen_command_ids.append(request.command_id)

    _send_and_check("follow_up", "Follow up", fill=True)
    _send_and_check("steer", "Steer", fill=True)
    _send_and_check("interrupt", "Interrupt", fill=True)
    _send_and_check("approve", "Approve", fill=False)
    _send_and_check("deny", "Deny", fill=False)
