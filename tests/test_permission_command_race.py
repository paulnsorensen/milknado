from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from milknado.domains.graph import MikadoGraph
from tests.graph_command_fixtures import NOW as _NOW
from tests.graph_command_fixtures import command as _command
from tests.graph_command_fixtures import owned_graph as _owned_graph


def test_pending_permission_accepts_one_decision(graph: MikadoGraph) -> None:
    node_id = _owned_graph(graph)
    approve = _command(
        node_id, command_id="approve-command", action="approve", permission_id="permission-1"
    )
    deny = _command(
        node_id, command_id="deny-command", action="deny", permission_id="permission-1"
    )

    first = graph.commands.admit(approve, now=_NOW)
    assert first.status == "queued"
    assert graph.commands.admit(approve, now=_NOW) == first
    rejected = graph.commands.admit(deny, now=_NOW)

    assert rejected.status == "rejected"
    assert rejected.detail == "permission already has a pending decision"
    assert [receipt.status for receipt in graph.commands.history("approve-command")] == ["queued"]
    assert [receipt.status for receipt in graph.commands.history("deny-command")] == ["rejected"]


def test_pending_permission_is_unique_across_concurrent_connections(tmp_path: Path) -> None:
    db_path = tmp_path / "permission-race.db"
    graph = MikadoGraph(db_path)
    try:
        node_id = _owned_graph(graph)
    finally:
        graph.close()
    barrier = Barrier(2)

    def admit(decision: str) -> tuple[str, str]:
        connection = MikadoGraph(db_path)
        try:
            command = _command(
                node_id, command_id=decision, action=decision, permission_id="permission-1"
            )
            _ = barrier.wait(timeout=5)
            return decision, connection.commands.admit(command, now=_NOW).status
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(admit, ("approve", "deny")))
    assert sorted(results.values()) == ["queued", "rejected"]
    graph = MikadoGraph(db_path)
    try:
        for decision, status in results.items():
            assert [receipt.status for receipt in graph.commands.history(decision)] == [status]
        assert len(graph.commands.pending("run-1", now=_NOW)) == 1
    finally:
        graph.close()
