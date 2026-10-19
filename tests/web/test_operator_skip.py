# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false
import sqlite3
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from milknado.domains.common import MikadoNode, NodeStatus, PluginMeta
from milknado.domains.graph import MikadoGraph
from milknado.web import LaunchToken, WebCommands, create_app
from milknado.web.commands import GraphEditCommands
from tests.web.support import FixtureSnapshotSource, headers


class _CommittedStatusRecorder:
    """Record the status another connection sees as each skip step lands."""

    def __init__(self, db_path: Path) -> None:
        self._db_path: Path = db_path
        self.seen: list[tuple[NodeStatus, str]] = []

    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(name="committed-status", version="0.1.0", description="")

    def on_node_status_change(
        self, node: MikadoNode, old_status: NodeStatus, new_status: NodeStatus
    ) -> None:
        del old_status
        reader = sqlite3.connect(self._db_path)
        try:
            row = reader.execute("SELECT status FROM nodes WHERE id = ?", (node.id,)).fetchone()
        finally:
            reader.close()
        self.seen.append((new_status, str(row[0])))


def test_skip_never_exposes_a_committed_pending_node_to_the_scheduler(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    recorder = _CommittedStatusRecorder(db_path)
    graph = MikadoGraph(db_path, plugins=[recorder])
    login = LaunchToken("test-token")
    commands = WebCommands(graph_edits=GraphEditCommands(graph, frozenset(), tmp_path))
    client = TestClient(
        create_app(FixtureSnapshotSource(), commands, login), base_url="http://127.0.0.1"
    )
    client.cookies.set(login.cookie_name, login.value)
    try:
        node_id = graph.add_node("strip").id
        graph.set_todo_status(node_id, NodeStatus.FAILED)
        recorder.seen.clear()

        response = client.post(f"/api/nodes/{node_id}/skip", json={}, headers=headers())

        assert response.status_code == 200
        assert response.json()["status"] == "done"
        assert [step for step, _ in recorder.seen] == [
            NodeStatus.PENDING,
            NodeStatus.RUNNING,
            NodeStatus.DONE,
        ]
        assert {committed for _, committed in recorder.seen} == {"failed"}
        stored = graph.get_node(node_id)
        assert stored is not None and stored.status is NodeStatus.DONE
    finally:
        graph.close()


def test_graph_skip_rejects_missing_and_unfailed_nodes(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node_id = graph.add_node("strip").id

        with pytest.raises(ValueError, match="Node 999 not found"):
            graph.skip_failed(999)
        with pytest.raises(ValueError, match=f"node {node_id} is pending; only a failed node"):
            graph.skip_failed(node_id)
        stored = graph.get_node(node_id)
        assert stored is not None and stored.status is NodeStatus.PENDING
    finally:
        graph.close()
