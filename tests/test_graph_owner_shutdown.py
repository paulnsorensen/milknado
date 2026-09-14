from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from milknado.domains.graph import MikadoGraph
from tests.graph_command_fixtures import NOW, command, owned_graph


def test_close_preserves_submitted_commands_and_replacement_owner(graph: MikadoGraph) -> None:
    node_id = owned_graph(graph)
    submitted = command(node_id, command_id="submitted")
    assert graph.commands.admit(submitted, now=NOW).status == "queued"
    assert graph.commands.submit(submitted, now=NOW).status == "submitted"
    assert graph.commands.admit(command(node_id), now=NOW).status == "queued"
    replacement = graph.commands.publish_capabilities(
        "run-1", node_id, "invoke-2", "owner-2", ("steer",), published_at=NOW
    )
    other = command(node_id, command_id="other", owner="owner-2", invocation="invoke-2")
    assert graph.commands.admit(other, now=NOW).status == "queued"

    graph.commands.close_owner("run-1", "owner-1", "invoke-1")
    graph.commands.close_owner("run-1", "owner-1", "invoke-1")

    assert graph.commands.capabilities("run-1") == replacement
    assert [r.status for r in graph.commands.history("command-1")] == ["queued", "rejected"]
    assert [r.status for r in graph.commands.history("submitted")] == ["queued", "submitted"]
    assert [r.status for r in graph.commands.history("other")] == ["queued"]


def test_empty_actions_do_not_close_an_invocation(graph: MikadoGraph) -> None:
    node_id = owned_graph(graph)
    assert graph.commands.admit(command(node_id), now=NOW).status == "queued"
    _ = graph.commands.publish_capabilities(
        "run-1", node_id, "invoke-1", "owner-1", (), published_at=NOW
    )
    assert [r.status for r in graph.commands.history("command-1")] == ["queued"]
    graph.commands.close_owner("run-1", "owner-1", "invoke-1")
    assert [r.status for r in graph.commands.history("command-1")] == ["queued", "rejected"]


def test_close_serializes_with_admission_on_another_connection(tmp_path: Path) -> None:
    db = tmp_path / "close-race.db"
    graph = MikadoGraph(db)
    try:
        node_id = owned_graph(graph)
    finally:
        graph.close()
    barrier = Barrier(2)

    def operation(close: bool) -> None:
        connection = MikadoGraph(db)
        try:
            _ = barrier.wait(timeout=5)
            if close:
                connection.commands.close_owner("run-1", "owner-1", "invoke-1")
            else:
                _ = connection.commands.admit(command(node_id), now=NOW)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(operation, (False, True))) == [None, None]
    graph = MikadoGraph(db)
    try:
        history = [r.status for r in graph.commands.history("command-1")]
        assert history in (["queued", "rejected"], ["rejected"])
        assert graph.commands.pending("run-1", now=NOW) == ()
        caps = graph.commands.capabilities("run-1")
        assert caps is not None
        assert caps.actions == ()
    finally:
        graph.close()
