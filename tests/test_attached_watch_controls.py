from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import pytest
from textual.widgets import Select

from milknado.app.watch import AttachedWatchSource, WatchSnapshotSource, graph_command_admitter
from milknado.app.watch_tui import WatchApp
from milknado.domains.common import SessionContext
from milknado.domains.graph import MikadoGraph
from milknado.domains.graph._session_persistence import view_session
from tests.graph_helpers import graph_conn

_NOW = "2026-09-12T12:00:00+00:00"


class _WorkerManager(Protocol):
    async def wait_for_complete(self) -> None: ...


def _wait_for_workers(app: WatchApp) -> _WorkerManager:
    return cast(_WorkerManager, app.workers)


def _owned_graph(db_path: Path) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(db_path)
    node = graph.add_node("Attached session")
    assert graph.claim_node(node.id, "run-1", now=_NOW)
    graph.runs.start("run-1", node.id, str(db_path.with_suffix(".log")), _NOW, 60)
    graph.sessions.start("run-1", SessionContext(family="claude", cwd=str(db_path.parent)))
    _ = graph.commands.publish_capabilities(
        "run-1", node.id, "turn-1", "owner-1", ("follow_up",), published_at=_NOW
    )
    return graph, node.id


@pytest.mark.asyncio
async def test_attached_watch_selects_action_and_queues_durable_command(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph, _ = _owned_graph(db_path)
    source = AttachedWatchSource(
        WatchSnapshotSource(tmp_path, db_path),
        graph_command_admitter(graph),
    )
    app = WatchApp(source, poll_interval=60.0, read_only=False)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            action = cast(Select[str], app.query_one("#session-action", Select))
            assert action.value == "follow_up"
            await pilot.press("i", *"send this", "enter")
            await _wait_for_workers(app).wait_for_complete()
            await pilot.pause()
        pending = graph.commands.pending("run-1")
        assert [(command.action, command.text) for command in pending] == [
            ("follow_up", "send this")
        ]
    finally:
        graph.close()


@pytest.mark.asyncio
async def test_readonly_watch_cannot_submit_and_terminal_caps_are_hidden(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph, _ = _owned_graph(db_path)
    source = WatchSnapshotSource(tmp_path, db_path)
    app = WatchApp(source, poll_interval=60.0)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("i", *"must not send", "enter")
            await pilot.pause()
            assert not graph.commands.pending("run-1")
        assert view_session(graph_conn(graph), "run-1", active=False).actions == ()
    finally:
        graph.close()
