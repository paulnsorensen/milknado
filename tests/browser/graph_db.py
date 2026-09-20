"""Browser harness pairing a real sqlite `MikadoGraph` writer with the
production `WatchSnapshotSource`/`PolledSnapshotSource` reader stack, exactly
as `milknado watch`/`cli/web.py` wire them (`src/milknado/cli/web.py`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from milknado.app.watch import WatchSnapshotSource
from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.graph import GoalReviewRequest, MikadoGraph
from milknado.web import LaunchToken, PolledSnapshotSource, WebCommands, create_app
from milknado.web.commands import GraphEditCommands
from tests.browser.conftest import BROWSER_TOKEN, BrowserServer

POLL_INTERVAL = 0.05


@dataclass
class GraphDbServer:
    """A running browser server backed by a real graph.db, with its node ids."""

    server: BrowserServer
    graph: MikadoGraph
    source: PolledSnapshotSource
    edit_node_id: int
    move_node_id: int
    target_parent_id: int
    archive_node_id: int
    review_id: int

    @property
    def login_url(self) -> str:
        return self.server.login_url


def _seed_nodes(graph: MikadoGraph) -> tuple[int, int, int, int, int]:
    edit_node = graph.add_node("Edit target")
    move_node = graph.add_node("Move target")
    target_parent = graph.add_node("Target parent")
    archive_node = graph.add_node("Archive target")
    # Only a DONE, non-blocking node is eligible for archive.
    graph.mark_running(archive_node.id)
    graph.mark_done(archive_node.id)
    goal = graph.add_node("Review goal", spec=NodeSpec(kind=NodeKind.GOAL))
    review = graph.request_goal_review(
        GoalReviewRequest(
            goal.id, "rev", "evidence for the change", "proposed change text", reviewer="reviewer"
        )
    )
    graph.register_controller_master()
    return edit_node.id, move_node.id, target_parent.id, archive_node.id, review.review_id


def _build_commands(graph: MikadoGraph, project_root: Path) -> WebCommands:
    return WebCommands(
        graph_edits=GraphEditCommands(graph, frozenset({"implement"}), project_root),
        review_decision=lambda request, *, decided_by: graph.decide_goal_review(
            request, decided_by=decided_by
        ),
    )


@pytest.fixture
def graph_db_server(tmp_path: Path) -> Iterator[GraphDbServer]:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    try:
        edit_id, move_id, target_id, archive_id, review_id = _seed_nodes(graph)
        commands = _build_commands(graph, tmp_path)
        watch_source = WatchSnapshotSource(tmp_path, db_path)
        source = PolledSnapshotSource(watch_source, interval=POLL_INTERVAL)
        source.start()
        login = LaunchToken(BROWSER_TOKEN)
        app = create_app(source, commands, login)
        server = BrowserServer(app=app, login=login)
        server.start()
        try:
            yield GraphDbServer(
                server=server,
                graph=graph,
                source=source,
                edit_node_id=edit_id,
                move_node_id=move_id,
                target_parent_id=target_id,
                archive_node_id=archive_id,
                review_id=review_id,
            )
        finally:
            server.stop()
            source.close()
    finally:
        graph.close()
