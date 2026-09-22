from __future__ import annotations

from dataclasses import replace
from typing import Protocol, cast

import pytest
from textual.events import Key
from textual.pilot import Pilot
from textual.widgets import TabbedContent, Tree

from milknado.app.graph_view import GraphTreeEntry
from milknado.app.run_source import NodeSnapshotRequest
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.domains.common import MikadoEdge
from milknado.domains.execution import RunLoopResult
from milknado.domains.graph import GraphSnapshot
from tests.graph_navigation_fixtures import (
    PagedSource,
    details_text,
    navigation_text,
    run_app,
    source,
    wait_for_requests,
)


class _MessageDriver(Protocol):
    def send_message(self, event: Key) -> None: ...


async def _walk_detail_pages(
    pilot: Pilot[RunLoopResult | None], app: ExecutionSnapshotApp, source_value: PagedSource
) -> None:
    assert "child-page-0" in details_text(app)
    assert "history-page-0" in details_text(app)
    await pilot.press("[")
    await pilot.pause()
    assert len(source_value.requests) == 1
    await pilot.press("]")
    await wait_for_requests(pilot, source_value, 2)
    assert "child-page-1" in details_text(app)
    assert "run-page-1" in details_text(app)
    assert "artifact-page-1" in details_text(app)
    assert "claim-page-1" in details_text(app)
    await pilot.press("]")
    await wait_for_requests(pilot, source_value, 3)
    assert "child-page-2" in details_text(app)
    assert "run-page-2" in details_text(app)
    assert "artifact-page-2" in details_text(app)
    assert "Related values page 3" in navigation_text(app)
    assert "] next" not in navigation_text(app)
    assert "claim-page-2" in details_text(app)
    await pilot.press("]")
    await pilot.pause()
    assert len(source_value.requests) == 3
    await pilot.press("[")
    await wait_for_requests(pilot, source_value, 4)
    await pilot.press("[")
    await wait_for_requests(pilot, source_value, 5)
    await pilot.press("[")
    await pilot.pause()
    assert len(source_value.requests) == 5


async def _walk_history_pages(
    pilot: Pilot[RunLoopResult | None], app: ExecutionSnapshotApp, source_value: PagedSource
) -> None:
    await pilot.press(")")
    await wait_for_requests(pilot, source_value, 6)
    assert "history-page-1" in details_text(app)
    assert "History page 2" in navigation_text(app)
    assert ") next" not in navigation_text(app)
    await pilot.press(")")
    await pilot.pause()
    assert len(source_value.requests) == 6
    await pilot.press("(")
    await wait_for_requests(pilot, source_value, 7)
    assert "history-page-0" in details_text(app)
    await pilot.press("(")
    await pilot.pause()
    assert len(source_value.requests) == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("run", "watch"))
async def test_pagination_keys_request_pages_and_render_values(kind: str) -> None:
    source_value = source()
    app = run_app(source_value, kind)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_requests(pilot, source_value, 1)
        app.query_one("#run-tabs", TabbedContent).active = "details"
        _ = app.query_one("#details-panel").focus()
        await pilot.pause()
        await _walk_detail_pages(pilot, app, source_value)
        await _walk_history_pages(pilot, app, source_value)

        first_generation = source_value.requests[0].request_generation
        assert source_value.requests == [
            NodeSnapshotRequest(1, first_generation, page=0, limit=50, session_event_page=0),
            NodeSnapshotRequest(1, first_generation + 1, page=1, limit=50, session_event_page=0),
            NodeSnapshotRequest(1, first_generation + 2, page=2, limit=50, session_event_page=0),
            NodeSnapshotRequest(1, first_generation + 3, page=1, limit=50, session_event_page=0),
            NodeSnapshotRequest(1, first_generation + 4, page=0, limit=50, session_event_page=0),
            NodeSnapshotRequest(1, first_generation + 5, page=0, limit=50, session_event_page=1),
            NodeSnapshotRequest(1, first_generation + 6, page=0, limit=50, session_event_page=0),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("run", "watch"))
async def test_stale_detail_response_is_not_displayed(kind: str) -> None:
    source_value = source()
    app = run_app(source_value, kind)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_requests(pilot, source_value, 1)
        app.query_one("#run-tabs", TabbedContent).active = "details"
        _ = app.query_one("#details-panel").focus()
        source_value.stale_next = True
        await pilot.press("]")
        await wait_for_requests(pilot, source_value, 2)
        first_generation = source_value.requests[0].request_generation
        assert source_value.requests[-1] == NodeSnapshotRequest(
            1, first_generation + 1, page=1, limit=50, session_event_page=0
        )
        assert "stale-page-1" not in details_text(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("run", "watch"))
async def test_receipt_pages_drive_detail_navigation(kind: str) -> None:
    source_value = source()
    source_value.receipts_only = True
    app = run_app(source_value, kind)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_requests(pilot, source_value, 1)
        app.query_one("#run-tabs", TabbedContent).active = "details"
        _ = app.query_one("#details-panel").focus()
        await pilot.pause()
        assert "receipt-text-page-0" in details_text(app)
        assert "status: queued" in details_text(app)
        assert "] next" in navigation_text(app)
        await pilot.press("]")
        await wait_for_requests(pilot, source_value, 2)
        first_generation = source_value.requests[0].request_generation
        assert source_value.requests[-1] == NodeSnapshotRequest(
            1, first_generation + 1, page=1, limit=50, session_event_page=0
        )
        assert "receipt-text-page-1" in details_text(app)
        assert "status: delivered" in details_text(app)
        assert "] next" not in navigation_text(app)


@pytest.mark.asyncio
async def test_queued_tree_keys_preserve_latest_cursor() -> None:
    source_value = source()
    current_graph = source_value.current.graph
    assert current_graph is not None
    sibling = current_graph.nodes[1]
    extras = (
        replace(sibling, id=3, description="third child"),
        replace(sibling, id=4, description="fourth child"),
    )
    source_value.current = replace(
        source_value.current,
        graph=GraphSnapshot(
            current_graph.nodes + extras,
            current_graph.edges + (MikadoEdge(1, 3), MikadoEdge(1, 4)),
            current_graph.root_ids,
        ),
    )
    app = run_app(source_value, "run")

    async with app.run_test(size=(120, 40)) as pilot:
        tree = cast(Tree[GraphTreeEntry], app.query_one("#graph-tree", Tree))
        _ = tree.root.expand_all()
        _ = tree.focus()
        await pilot.pause()
        driver = cast(_MessageDriver, vars(app)["_driver"])
        for _ in range(3):
            event = Key("down", None)
            _ = event.set_sender(app)
            driver.send_message(event)
        await pilot.pause()

        assert app.selected_node_id == 4
        assert tree.cursor_node is not None
        assert tree.cursor_node.data is not None
        assert tree.cursor_node.data.node_id == 4
        assert tree.has_focus
