from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

import pytest
from rich.text import Text
from textual.widgets import Footer, Header, Input, Static

from milknado.domains.common import SessionInput
from tests.graph_navigation_fixtures import PagedSource, run_app, source, wait_for_requests


class _WorkerWaiter(Protocol):
    async def wait_for_complete(self) -> None: ...


@dataclass
class _RecordingSource(PagedSource):
    submissions: list[tuple[str, SessionInput]] = field(default_factory=list)

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        self.submissions.append((run_id, command))
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("run", "watch"))
async def test_minimum_layout_replaces_workspace_and_restores_visible_controls(kind: str) -> None:
    source_value = source()
    app = run_app(source_value, kind)
    async with app.run_test(size=(40, 15)) as pilot:
        await wait_for_requests(pilot, source_value, 1)
        fallback = app.query_one("#minimum-fallback", Static)
        assert fallback.region.area > 0
        assert fallback.region.x + fallback.region.width <= app.size.width
        assert cast(Text, fallback.render()).plain == (
            "Terminal too small for session controls. Resize to at least 60x18.\nq quit · ? help"
        )
        for widget in (
            app.query_one("#workspace"),
            app.query_one("#events"),
            app.query_one(Header),
            app.query_one(Footer),
        ):
            assert widget.region.area == 0
        await pilot.press("e", "j", "i", "x", "]", ")", "enter", "tab")
        await pilot.pause()
        assert (app.selected_node_id, app.selected_run_id) == (1, "run-1")
        assert app.minimum is True
        assert app.route == "list"
        for width, height in ((80, 24), (120, 40)):
            await pilot.resize_terminal(width=width, height=height)
            await pilot.pause()
            assert app.minimum is False
            assert fallback.region.area == 0
            assert app.query_one("#workspace").region.area > 0
            assert app.query_one(Footer).region.area > 0
            await pilot.press("e")
            await pilot.pause()
            assert app.screen.focused is app.query_one("#events")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("run", "watch"))
async def test_minimum_help_lists_only_available_controls(kind: str) -> None:
    app = run_app(source(), kind)
    async with app.run_test(size=(40, 15)) as pilot:
        await pilot.pause()
        await pilot.press("?")
        await pilot.pause()
        assert app.screen.is_modal
        body = cast(Text, app.screen.query_one("#help-overlay", Static).render()).plain
        assert "60x18" in body
        assert "quit" in body
        for unavailable in ("session input", "changed files", "next run", "related page"):
            assert unavailable not in body.lower()
        await pilot.press("escape")
        await pilot.pause()
        assert not app.screen.is_modal
        assert app.query_one("#minimum-fallback").region.area > 0


@pytest.mark.asyncio
async def test_minimum_resize_preserves_draft_without_hidden_submission() -> None:
    source_value = _RecordingSource(source().current, [])
    app = run_app(source_value, "run")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_requests(pilot, source_value, 1)
        await pilot.press("i")
        await pilot.pause()
        editor = app.query_one("#session-input", Input)
        assert app.screen.focused is editor
        await pilot.press(*"keep this draft")
        assert app.session_draft == "keep this draft"
        await pilot.resize_terminal(width=40, height=15)
        await pilot.pause()
        await pilot.press("enter", "tab", "enter", "i", "enter")
        await cast(_WorkerWaiter, app.workers).wait_for_complete()
        await pilot.pause()
        assert source_value.submissions == []
        assert app.session_draft == "keep this draft"
        assert editor.region.area == 0
        for width, height in ((80, 24), (120, 40)):
            await pilot.resize_terminal(width=width, height=height)
            await pilot.pause()
            assert app.screen.focused is editor
            assert editor.region.area > 0
            assert editor.value == "keep this draft"
        await pilot.press("enter")
        await cast(_WorkerWaiter, app.workers).wait_for_complete()
        await pilot.pause()
        assert len(source_value.submissions) == 1
        run_id, command = source_value.submissions[0]
        assert (run_id, command.action, command.text) == ("run-1", "steer", "keep this draft")
        assert command.command_id


@pytest.mark.asyncio
async def test_minimum_run_quit_keeps_confirmation_and_cancel_behavior() -> None:
    app = run_app(source(), "run")
    async with app.run_test(size=(40, 15)) as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert app.screen.id == "confirmation-screen"
        assert app.screen.query_one("#confirmation-overlay").region.area > 0
        overlay = app.screen.query_one("#confirmation-overlay")
        assert overlay.region.x + overlay.region.width <= app.screen.size.width
        await pilot.press("n")
        await pilot.pause()
        assert not app.screen.is_modal
        assert app.query_one("#minimum-fallback").region.area > 0
