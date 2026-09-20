"""AC-3: a published snapshot updates the graph, roster, and console live."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from milknado.app.run_source import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
)
from milknado.domains.common import MikadoEdge, MikadoNode, NodeKind
from milknado.domains.graph import GraphSnapshot
from tests.browser.conftest import (
    FIXTURE_NODE_DESCRIPTION,
    BrowserServer,
    BrowserSnapshotSource,
)

pytestmark = pytest.mark.browser

NEW_NODE_DESCRIPTION = "Freshly published task"
NEW_RUN_DESCRIPTION = "Bake the new task"
NEW_EVENT_LINE = "a live console line"


def _published_snapshot() -> ExecutionSnapshot:
    root = MikadoNode(id=1, description=FIXTURE_NODE_DESCRIPTION, kind=NodeKind.GOAL)
    child = MikadoNode(id=2, description=NEW_NODE_DESCRIPTION, kind=NodeKind.TASK, parent_id=1)
    graph = GraphSnapshot(nodes=(root, child), edges=(MikadoEdge(1, 2),), root_ids=(1,))
    run = ActiveRunSnapshot(
        run_id="run-1",
        node_id=2,
        description=NEW_RUN_DESCRIPTION,
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
    return ExecutionSnapshot(
        goal="Tracer fixture goal",
        active_runs=(run,),
        terminal_runs=(),
        completed=0,
        failed=0,
        stopped=0,
        available=1,
        event_lines=(NEW_EVENT_LINE,),
        listener_errors=(),
        graph=graph,
        node=None,
    )


def test_published_snapshot_updates_the_page_without_navigating(
    page: Page, browser_server: BrowserServer, browser_source: BrowserSnapshotSource
) -> None:
    _ = page.goto(browser_server.login_url)
    expect(page.get_by_text(FIXTURE_NODE_DESCRIPTION)).to_be_visible()
    url_before = page.url

    browser_source.publish(_published_snapshot())

    expect(page.get_by_text(NEW_NODE_DESCRIPTION)).to_be_visible()
    expect(page.get_by_text(NEW_RUN_DESCRIPTION)).to_be_visible()
    expect(page.get_by_text(NEW_EVENT_LINE)).to_be_visible()
    assert page.url == url_before
