# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false
from pathlib import Path

from starlette.testclient import TestClient

from milknado.domains.common import NodeStatus
from milknado.domains.graph import MikadoGraph
from milknado.web import LaunchToken, WebCommands, create_app
from milknado.web.commands import GraphEditCommands
from tests.web.support import FixtureSnapshotSource, headers


def _client(tmp_path: Path) -> tuple[TestClient, MikadoGraph]:
    graph = MikadoGraph(tmp_path / "graph.db")
    commands = WebCommands(
        graph_edits=GraphEditCommands(graph, frozenset({"implement"}), tmp_path)
    )
    login = LaunchToken("test-token")
    client = TestClient(
        create_app(FixtureSnapshotSource(), commands, login), base_url="http://127.0.0.1"
    )
    client.cookies.set(login.cookie_name, login.value)
    return client, graph


def test_node_mutations_apply_to_graph(tmp_path: Path) -> None:
    client, graph = _client(tmp_path)
    try:
        response = client.post("/api/nodes", json={"description": "root"}, headers=headers())
        assert response.status_code == 200
        malformed = client.post("/api/nodes", content=b"{", headers=headers())
        assert malformed.status_code == 400
        node_id = response.json()["id"]
        child = client.post(
            "/api/nodes", json={"description": "child", "parent_id": node_id}, headers=headers()
        )
        assert child.status_code == 200
        child_id = child.json()["id"]
        assert (
            client.patch(
                f"/api/nodes/{child_id}", json={"description": "edited"}, headers=headers()
            ).status_code
            == 200
        )
        rejected = client.post(
            f"/api/nodes/{node_id}/move", json={"new_parent_id": child_id}, headers=headers()
        )
        assert rejected.status_code == 409
        assert (
            client.post(
                f"/api/nodes/{child_id}/move", json={"new_parent_id": None}, headers=headers()
            ).status_code
            == 200
        )
        graph.set_todo_status(child_id, NodeStatus.DONE)
        assert client.post(f"/api/nodes/{child_id}/archive", headers=headers()).status_code == 200
        updated = graph.get_node(child_id)
        assert updated is not None
        assert updated.description == "edited"
        assert updated.archived_at is not None
    finally:
        graph.close()


def test_node_routes_reject_bad_requests(tmp_path: Path) -> None:
    client, graph = _client(tmp_path)
    try:
        created = client.post("/api/nodes", json={"description": "node"}, headers=headers())
        node_id = created.json()["id"]
        assert (
            client.post(
                "/api/nodes",
                json={"description": "bad", "artifact": "../escape"},
                headers=headers(),
            ).status_code
            == 409
        )
        assert client.patch(f"/api/nodes/{node_id}", json={}, headers=headers()).status_code == 409
        assert (
            client.patch(f"/api/nodes/{node_id}", content=b"{", headers=headers()).status_code
            == 400
        )
        assert (
            client.patch(
                f"/api/nodes/{node_id}", json={"files": []}, headers=headers()
            ).status_code
            == 200
        )
        assert (
            client.patch(
                "/api/nodes/999", json={"description": "missing"}, headers=headers()
            ).status_code
            == 409
        )
    finally:
        graph.close()
