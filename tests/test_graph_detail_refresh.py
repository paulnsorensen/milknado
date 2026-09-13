from __future__ import annotations

from typing import Protocol, cast

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, TabbedContent

from milknado.app.graph_pagination import related_pages
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.domains.execution import RunLoopResult
from tests.graph_navigation_fixtures import (
    PagedSource,
    details_text,
    navigation_text,
    run_app,
    source,
    wait_for_requests,
)


def test_missing_detail_has_no_related_pages() -> None:
    assert related_pages(None) == ()


class _WorkerWaiter(Protocol):
    async def wait_for_complete(self) -> None: ...


async def _wait_for_started(pilot: Pilot[RunLoopResult | None], source_value: PagedSource) -> None:
    for _ in range(40):
        await pilot.pause()
        if source_value.started.is_set():
            return
    raise AssertionError("detail request did not reach the controlled source")


async def _prepare_detail_state(
    pilot: Pilot[RunLoopResult | None],
    app: ExecutionSnapshotApp,
    source_value: PagedSource,
) -> object:
    assert len(app.runs()) == 1
    app.query_one("#run-tabs", TabbedContent).active = "details"
    details_panel = app.query_one("#details-panel")
    _ = details_panel.focus()
    await pilot.press("]")
    await wait_for_requests(pilot, source_value, 2)
    await pilot.press(")")
    await wait_for_requests(pilot, source_value, 3)
    assert app.detail_page == 1
    assert app.session_event_page == 1
    assert "child-page-1" in details_text(app)
    assert "history-page-1" in details_text(app)
    assert "History page 2" in navigation_text(app)
    if app.read_only is False:
        session_input = app.query_one("#session-input", Input)
        _ = session_input.focus()
        await pilot.press(*"draft text")
        assert session_input.value == "draft text"
        assert app.session_draft == "draft text"
        _ = details_panel.focus()
    return details_panel


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("run", "watch"))
async def test_snapshot_refresh_preserves_visible_state_until_fresh_detail(
    kind: str,
) -> None:
    source_value = source()
    app = run_app(source_value, kind)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_requests(pilot, source_value, 1)
        details_panel = await _prepare_detail_state(pilot, app, source_value)

        source_value.block_next = True
        source_value.detail_marker = "old-"
        app.show_snapshot(source_value.current)
        await _wait_for_started(pilot, source_value)
        assert app.detail_page == 1
        assert app.session_event_page == 1
        assert app.screen.focused is details_panel
        assert app.session_draft == ("draft text" if kind == "run" else "")
        assert "child-page-1" in details_text(app)
        assert "history-page-1" in details_text(app)

        source_value.detail_marker = "fresh-"
        app.show_snapshot(source_value.current)
        assert "child-page-1" in details_text(app)
        source_value.release.set()
        for _ in range(40):
            await pilot.pause()
            if "fresh-child-page-1" in details_text(app):
                break
        assert source_value.requests[-1].page == 1
        assert source_value.requests[-1].session_event_page == 1
        assert app.detail_page == 1
        assert app.session_event_page == 1
        assert app.screen.focused is details_panel
        assert app.session_draft == ("draft text" if kind == "run" else "")
        assert "fresh-child-page-1" in details_text(app)
        assert "fresh-run-page-1" in details_text(app)
        assert "fresh-review-page-1" in details_text(app)
        assert "fresh-history-page-1" in details_text(app)
        assert "fresh-artifact-page-1" in details_text(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("run", "watch"))
async def test_detail_response_fences_reject_old_generation_and_node(
    kind: str,
) -> None:
    source_value = source()
    app = run_app(source_value, kind)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_requests(pilot, source_value, 1)
        app.query_one("#run-tabs", TabbedContent).active = "details"
        _ = app.query_one("#details-panel").focus()
        visible = details_text(app)

        source_value.response_generation_delta = -1
        source_value.detail_marker = "wrong-generation-"
        app.show_snapshot(source_value.current)
        await wait_for_requests(pilot, source_value, 2)
        await cast(_WorkerWaiter, app.workers).wait_for_complete()
        assert details_text(app) == visible

        source_value.response_node_id = 999
        source_value.detail_marker = "wrong-node-"
        app.show_snapshot(source_value.current)
        await wait_for_requests(pilot, source_value, 3)
        await cast(_WorkerWaiter, app.workers).wait_for_complete()
        assert details_text(app) == visible
