# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false
import os
import time
from pathlib import Path
from threading import Thread

import pytest
from starlette.testclient import TestClient

from milknado.domains.common import NodeStatus
from milknado.domains.common.config import MilknadoConfig
from milknado.domains.graph import MikadoGraph
from milknado.web import LaunchToken, WebCommands, build_capabilities, create_app
from milknado.web.commands import GraphEditCommands, MetaharnessInfo
from milknado.web.metaharness import build_metaharness_info
from tests.web.support import FixtureSnapshotSource, headers


class FakeScheduling:
    def __init__(self) -> None:
        self.paused: bool = False

    def set_scheduling_paused(self, paused: bool) -> None:
        self.paused = paused

    def scheduling_paused(self) -> bool:
        return self.paused


def _client(commands: WebCommands) -> TestClient:
    login = LaunchToken("test-token")
    client = TestClient(
        create_app(FixtureSnapshotSource(), commands, login), base_url="http://127.0.0.1"
    )
    client.cookies.set(login.cookie_name, login.value)
    return client


def test_metaharness_reports_project_flavors_and_knowledge(tmp_path: Path) -> None:
    project = tmp_path / "acme"
    wiki = project / ".hallouminate" / "wiki"
    wiki.mkdir(parents=True)
    (wiki / "design.md").write_text("# Design\n\nWhy it works.", encoding="utf-8")
    os.utime(wiki / "design.md", (0, 0))
    info = MetaharnessInfo(project, 3, {"implement": ("just test",), "spec": ()})
    scheduling = FakeScheduling()
    client = _client(WebCommands(metaharness=info, scheduling=scheduling))

    body = client.get("/api/metaharness").json()

    assert body == {
        "project": {"name": "acme"},
        "concurrency": 3,
        "scheduling": {"available": True, "paused": False, "reason": None},
        "flavors": {"implement": {"quality_gates": ["just test"]}, "spec": {"quality_gates": []}},
        "knowledge": [
            {
                "path": "design.md",
                "title": "Design",
                "summary": "Why it works.",
                "updated_at": "1970-01-01T00:00:00+00:00",
            }
        ],
    }


def test_metaharness_observer_has_no_pause_and_empty_knowledge(tmp_path: Path) -> None:
    client = _client(WebCommands(metaharness=MetaharnessInfo(tmp_path, None, {})))

    body = client.get("/api/metaharness").json()

    assert body["concurrency"] is None
    assert body["knowledge"] == []
    assert body["scheduling"] == {
        "available": False,
        "paused": False,
        "reason": "Pausing the run loop is unavailable.",
    }


def test_metaharness_unavailable_returns_409() -> None:
    response = _client(WebCommands()).get("/api/metaharness")

    assert response.status_code == 409
    assert response.json() == {"reason": "Metaharness info is unavailable."}


def test_pause_and_resume_drive_the_scheduling_control() -> None:
    scheduling = FakeScheduling()
    commands = WebCommands(scheduling=scheduling)
    client = _client(commands)

    paused = client.post("/api/scheduling/pause", headers=headers())
    assert paused.json() == {"paused": True}
    assert scheduling.paused
    resumed = client.post("/api/scheduling/resume", headers=headers())
    assert resumed.json() == {"paused": False}
    assert not scheduling.paused
    assert build_capabilities(commands)["scheduling_pause"] == {"available": True, "reason": None}


@pytest.mark.parametrize("action", ["pause", "resume"])
def test_pause_resume_unavailable_returns_409(action: str) -> None:
    response = _client(WebCommands()).post(f"/api/scheduling/{action}", headers=headers())

    assert response.status_code == 409
    assert response.json() == {"reason": "Pausing the run loop is unavailable."}


def test_build_metaharness_info_resolves_flavor_gates(tmp_path: Path) -> None:
    config = MilknadoConfig(project_root=tmp_path, concurrency_limit=6)

    info = build_metaharness_info(config, tmp_path)

    assert info.concurrency == 6
    assert info.project_root == tmp_path
    assert set(info.flavor_gates) == set(config.flavor_registry)
    assert all(isinstance(gates, tuple) for gates in info.flavor_gates.values())


def _graph_client(tmp_path: Path) -> tuple[TestClient, MikadoGraph]:
    graph = MikadoGraph(tmp_path / "graph.db")
    commands = WebCommands(graph_edits=GraphEditCommands(graph, frozenset(), tmp_path))
    return _client(commands), graph


def test_note_appends_operator_note_to_description(tmp_path: Path) -> None:
    client, graph = _graph_client(tmp_path)
    try:
        node_id = graph.add_node("fix it").id

        response = client.post(
            f"/api/nodes/{node_id}/note", json={"text": "try X"}, headers=headers()
        )

        assert response.status_code == 200
        expected = "fix it\n\nOperator note: try X"
        assert response.json()["description"] == expected
        stored = graph.get_node(node_id)
        assert stored is not None and stored.description == expected
    finally:
        graph.close()


def test_concurrent_notes_keep_every_operator_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, graph = _graph_client(tmp_path)
    try:
        node_id = graph.add_node("n").id
        read_node = graph.get_node

        def slow_read(target: int):
            node = read_node(target)
            time.sleep(0.05)
            return node

        monkeypatch.setattr(graph, "get_node", slow_read)
        codes: list[int] = []

        def post_note(text: str) -> None:
            url = f"/api/nodes/{node_id}/note"
            codes.append(client.post(url, json={"text": text}, headers=headers()).status_code)

        writers = [Thread(target=post_note, args=(text,)) for text in ("a", "b")]
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join()

        stored = read_node(node_id)
        assert codes == [200, 200]
        assert stored is not None
        assert "Operator note: a" in stored.description
        assert "Operator note: b" in stored.description
    finally:
        graph.close()


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   "}, {"text": "x" * 4001}, {"text": 5}, {}, {"text": "a", "x": 1}],
)
def test_note_rejects_invalid_bodies(tmp_path: Path, payload: dict[str, object]) -> None:
    client, graph = _graph_client(tmp_path)
    try:
        node_id = graph.add_node("n").id

        response = client.post(f"/api/nodes/{node_id}/note", json=payload, headers=headers())

        assert response.status_code == 400
        for raw in (b"{", b"\xff"):
            malformed = client.post(f"/api/nodes/{node_id}/note", content=raw, headers=headers())
            assert malformed.status_code == 400
    finally:
        graph.close()


def test_note_and_retry_report_unknown_node_and_missing_graph(tmp_path: Path) -> None:
    client, graph = _graph_client(tmp_path)
    try:
        for route, body in (
            ("note", {"text": "a"}),
            ("retry", {}),
            ("skip", {}),
        ):
            response = client.post(f"/api/nodes/999/{route}", json=body, headers=headers())
            assert response.status_code == 409
            assert response.json() == {"error": "node 999 not found"}
            bare = _client(WebCommands()).post(
                f"/api/nodes/1/{route}", json=body, headers=headers()
            )
            assert bare.status_code == 409
            assert bare.json() == {"error": "Graph edits are unavailable."}
    finally:
        graph.close()


@pytest.mark.parametrize("status", [NodeStatus.FAILED, NodeStatus.BLOCKED])
def test_retry_requeues_failed_or_blocked_node_with_note(
    tmp_path: Path, status: NodeStatus
) -> None:
    client, graph = _graph_client(tmp_path)
    try:
        node_id = graph.add_node("flaky").id
        graph.set_todo_status(node_id, status)

        response = client.post(
            f"/api/nodes/{node_id}/retry", json={"note": "use Y"}, headers=headers()
        )

        assert response.status_code == 200
        assert response.json()["status"] == "pending"
        assert response.json()["description"] == "flaky\n\nOperator note: use Y"
    finally:
        graph.close()


def test_retry_without_note_keeps_description_and_rejects_other_statuses(tmp_path: Path) -> None:
    client, graph = _graph_client(tmp_path)
    try:
        node_id = graph.add_node("flaky").id
        rejected = client.post(f"/api/nodes/{node_id}/retry", json={}, headers=headers())
        assert rejected.status_code == 409
        assert "only failed or blocked" in rejected.json()["error"]
        graph.set_todo_status(node_id, NodeStatus.FAILED)

        retried = client.post(
            f"/api/nodes/{node_id}/retry", json={"note": None}, headers=headers()
        )

        assert retried.status_code == 200
        assert retried.json()["status"] == "pending"
        assert retried.json()["description"] == "flaky"
        assert (
            client.post(
                f"/api/nodes/{node_id}/retry", json={"note": ""}, headers=headers()
            ).status_code
            == 400
        )
    finally:
        graph.close()


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        (None, "Skipped by the operator."),
        ("covered elsewhere", "Skipped by the operator. covered elsewhere"),
    ],
)
def test_skip_marks_failed_node_done_with_operator_note(
    tmp_path: Path, note: str | None, expected: str
) -> None:
    client, graph = _graph_client(tmp_path)
    try:
        node_id = graph.add_node("strip").id
        graph.set_todo_status(node_id, NodeStatus.FAILED)

        response = client.post(
            f"/api/nodes/{node_id}/skip", json={"note": note}, headers=headers()
        )

        assert response.status_code == 200
        assert response.json()["status"] == "done"
        assert response.json()["description"] == f"strip\n\nOperator note: {expected}"
    finally:
        graph.close()


def test_skip_rejects_a_node_that_has_not_failed(tmp_path: Path) -> None:
    client, graph = _graph_client(tmp_path)
    try:
        node_id = graph.add_node("strip").id

        response = client.post(f"/api/nodes/{node_id}/skip", json={}, headers=headers())

        assert response.status_code == 409
        assert response.json() == {"error": f"node {node_id} is pending; only a failed node skips"}
        assert (
            client.post(f"/api/nodes/{node_id}/skip", content=b"{", headers=headers()).status_code
            == 400
        )
    finally:
        graph.close()
