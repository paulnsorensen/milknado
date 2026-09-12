from __future__ import annotations

from typing import cast

import pytest

from milknado.domains.common import RunResult
from milknado.domains.graph import GraphCommand, MikadoGraph
from tests.graph_command_fixtures import NOW, command, owned_graph

LATER = "2026-09-12T12:05:00+00:00"


def test_admission_rejects_bad_time_without_durable_mutation(graph: MikadoGraph) -> None:
    node_id = owned_graph(graph)
    invalid = command(node_id, expires_at="not-a-timestamp")
    with pytest.raises(ValueError, match="ISO-8601"):
        _ = graph.commands.admit(invalid, now=NOW)
    assert graph.commands.command(invalid.command_id) is None
    with pytest.raises(ValueError, match="timezone"):
        _ = graph.commands.admit(command(node_id, command_id="no-zone"), now="2026-09-12T12:00:00")
    assert graph.commands.history("no-zone") == ()


@pytest.mark.parametrize("identifier", ["", "x" * 257])
def test_admission_rejects_identifier_boundaries(graph: MikadoGraph, identifier: str) -> None:
    node_id = owned_graph(graph)
    invalid = command(node_id, command_id=identifier)
    with pytest.raises(ValueError, match="command_id"):
        _ = graph.commands.admit(invalid, now=NOW)
    assert graph.commands.command(identifier) is None


def test_admission_rejects_oversized_utf8_text(graph: MikadoGraph) -> None:
    node_id = owned_graph(graph)
    invalid = command(node_id, text="é" * (65536 // 2 + 1))
    with pytest.raises(ValueError, match="65536"):
        _ = graph.commands.admit(invalid, now=NOW)
    assert graph.commands.history(invalid.command_id) == ()


def test_capability_validation_preserves_current_snapshot(graph: MikadoGraph) -> None:
    node_id = owned_graph(graph)
    original = graph.commands.capabilities("run-1")
    assert original is not None
    invalid = (
        (
            ("duplicate action", ("steer", "steer"), ("permission-1",), node_id),
            "owner capabilities must not contain duplicates",
        ),
        (
            ("duplicate permission", ("steer",), ("permission-1", "permission-1"), node_id),
            "owner capabilities must not contain duplicates",
        ),
        (
            ("wrong node", ("steer",), ("permission-1",), node_id + 1),
            "owner capabilities do not match the run node",
        ),
        (
            ("invalid action", ("unknown",), ("permission-1",), node_id),
            "owner capabilities contain an invalid node or action",
        ),
    )
    for case, expected_error in invalid:
        _, actions, permissions, invalid_node = case
        with pytest.raises(ValueError, match=expected_error):
            _ = graph.commands.publish_capabilities(
                "run-1",
                invalid_node,
                "invoke-2",
                "owner-2",
                actions,
                permissions,
                published_at=LATER,
            )
        assert graph.commands.capabilities("run-1") == original


def test_capability_publication_rejects_terminal_run(graph: MikadoGraph) -> None:
    node_id = owned_graph(graph)
    original = graph.commands.capabilities("run-1")
    assert original is not None
    graph.runs.finish(
        "run-1",
        RunResult(
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at=LATER,
        ),
    )
    with pytest.raises(ValueError, match="running"):
        _ = graph.commands.publish_capabilities(
            "run-1", node_id, "invoke-2", "owner-2", ("steer",), published_at=LATER
        )
    assert graph.commands.capabilities("run-1") == original


def test_public_admission_rejects_invalid_node_action_and_stale_run(
    graph: MikadoGraph,
) -> None:
    node_id = owned_graph(graph)
    invalid_run = GraphCommand(
        command_id="missing-run",
        node_id=node_id,
        run_id="missing",
        invocation_id="invoke-1",
        owner_incarnation="owner-1",
        action="steer",
        text="redirect",
        expires_at=LATER,
    )
    invalid = (
        (command(0), "node_id must be positive"),
        (command(node_id, action=cast(str, "unknown")), "unsupported session action"),
    )
    for value, expected_error in invalid:
        with pytest.raises(ValueError, match=expected_error):
            _ = graph.commands.admit(value, now=NOW)
        assert graph.commands.command(value.command_id) is None
        assert graph.commands.history(value.command_id) == ()
    rejected = graph.commands.admit(invalid_run, now=NOW)
    assert rejected.status == "rejected"
    assert graph.commands.pending("run-1") == ()
