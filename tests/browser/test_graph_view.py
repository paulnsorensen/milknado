"""AC-13: graph-view controls change the visible node set, never a command."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
from playwright.sync_api import Page, expect

from milknado.app.run_source import ExecutionSnapshot
from milknado.domains.common import MikadoEdge, MikadoNode, NodeKind
from milknado.domains.graph import GraphSnapshot
from milknado.web import LaunchToken, create_app
from tests.browser.conftest import (
    BROWSER_TOKEN,
    BrowserServer,
    RecordingCommands,
    open_app,
    owner_web_commands,
)
from tests.browser.conftest import BrowserSnapshotSource as SnapshotSource

pytestmark = pytest.mark.browser

ROOT_ID = 1
CHAIN_COUNT = 8
LEVELS = 5
TOTAL_NODES = 1 + CHAIN_COUNT * LEVELS
# Any node's ancestors + itself + descendants span its whole root-to-leaf chain.
CHAIN_LENGTH = 1 + LEVELS


def _chain_node_id(chain: int, level: int) -> int:
    return (chain + 1) * 10 + level


def _chain_graph_snapshot() -> ExecutionSnapshot:
    """41 nodes as 8 root-to-leaf chains of 5, none of them at-risk of an
    auto level-of-detail downgrade (each depth row holds exactly 8 nodes).
    """
    nodes: list[MikadoNode] = [MikadoNode(id=ROOT_ID, description="Root goal", kind=NodeKind.GOAL)]
    for level in range(1, LEVELS + 1):
        for chain in range(CHAIN_COUNT):
            parent_id = ROOT_ID if level == 1 else _chain_node_id(chain, level - 1)
            nodes.append(
                MikadoNode(
                    id=_chain_node_id(chain, level),
                    description=f"Chain {chain} level {level} has a balanced title",
                    kind=NodeKind.TASK,
                    parent_id=parent_id,
                )
            )
    edges = tuple(MikadoEdge(n.parent_id, n.id) for n in nodes if n.parent_id is not None)
    graph = GraphSnapshot(nodes=tuple(nodes), edges=edges, root_ids=(ROOT_ID,))
    return ExecutionSnapshot(
        goal="Chain fixture goal",
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
def graph_view_server() -> Iterator[tuple[BrowserServer, RecordingCommands]]:
    login = LaunchToken(BROWSER_TOKEN)
    source = SnapshotSource(snapshot=_chain_graph_snapshot())
    commands, recorder = owner_web_commands()
    app = create_app(source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server, recorder
    server.stop()


def _open(page: Page, server: BrowserServer) -> None:
    open_app(page, server.login_url, page.locator(".mk-graph-node").first)


def _assert_no_commands(recorder: RecordingCommands) -> None:
    assert recorder.session_input_calls == []
    assert recorder.cancel_calls == []
    assert recorder.force_stop_calls == []
    assert recorder.stop_scheduling_calls == 0
    assert recorder.review_decision_calls == []


def test_search_jump_focuses_the_matching_subtree(
    page: Page, graph_view_server: tuple[BrowserServer, RecordingCommands]
) -> None:
    server, recorder = graph_view_server
    _open(page, server)
    title = "Chain 3 level 3 has a balanced title"

    expect(page.locator(".mk-graph-node.is-faded")).to_have_count(0)
    page.get_by_label("Jump to node").fill(title)
    page.get_by_role("option", name=title).click()

    expect(page.locator(".mk-graph-node.is-faded")).to_have_count(TOTAL_NODES - CHAIN_LENGTH)
    _assert_no_commands(recorder)


def test_compact_graph_cards_do_not_intersect(
    page: Page, graph_view_server: tuple[BrowserServer, RecordingCommands]
) -> None:
    server, recorder = graph_view_server
    _open(page, server)

    boxes = cast(
        list[dict[str, float]],
        page.locator(".mk-graph-node").evaluate_all(
            """(nodes) => nodes.map((node) => {
                const box = node.getBoundingClientRect();
                return {left: box.left, right: box.right, top: box.top, bottom: box.bottom};
            })"""
        ),
    )
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            assert (
                first["right"] <= second["left"]
                or second["right"] <= first["left"]
                or first["bottom"] <= second["top"]
                or second["bottom"] <= first["top"]
            ), f"graph cards intersect: {first} and {second}"
    _assert_no_commands(recorder)


def test_ready_filter_hides_every_node(
    page: Page, graph_view_server: tuple[BrowserServer, RecordingCommands]
) -> None:
    server, recorder = graph_view_server
    _open(page, server)
    filter_group = page.get_by_role("group", name="Filter")

    expect(page.locator(".mk-graph-node")).to_have_count(TOTAL_NODES)
    filter_group.get_by_role("button", name="Ready", exact=True).click()
    expect(page.locator(".mk-graph-node")).to_have_count(0)
    filter_group.get_by_role("button", name="Ready", exact=True).click()

    expect(page.locator(".mk-graph-node")).to_have_count(TOTAL_NODES)
    _assert_no_commands(recorder)


def test_focus_fades_nodes_outside_the_selected_subtree(
    page: Page, graph_view_server: tuple[BrowserServer, RecordingCommands]
) -> None:
    server, recorder = graph_view_server
    _open(page, server)
    title = "Chain 5 level 2 has a balanced title"

    page.get_by_role("button", name=f"pending {title}", exact=True).click()
    page.get_by_role("button", name="Focus", exact=True).click()

    expect(page.locator(".mk-graph-node.is-faded")).to_have_count(TOTAL_NODES - CHAIN_LENGTH)
    _assert_no_commands(recorder)


def test_level_of_detail_switches_nodes_to_dots(
    page: Page, graph_view_server: tuple[BrowserServer, RecordingCommands]
) -> None:
    server, recorder = graph_view_server
    _open(page, server)
    node_style = page.get_by_role("group", name="Node style")

    expect(page.locator(".mk-node")).to_have_count(TOTAL_NODES)
    node_style.get_by_role("button", name="dots", exact=True).click()

    expect(page.locator(".mk-node")).to_have_count(0)
    expect(page.locator(".mk-dotnode")).to_have_count(TOTAL_NODES)
    _assert_no_commands(recorder)


def test_wide_canvas_does_not_mount_placeholder_minimap(
    page: Page, graph_view_server: tuple[BrowserServer, RecordingCommands]
) -> None:
    server, recorder = graph_view_server
    _open(page, server)

    expect(page.locator('[aria-label="Overview of the graph"]')).to_have_count(0)
    _assert_no_commands(recorder)
