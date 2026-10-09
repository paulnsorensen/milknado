from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.coordinator import ProviderBinding
from milknado.domains.coordinator.persistence import (
    bind_provider_session,
    link_entity,
    start_coordinator,
)
from milknado.domains.coordinator.recovery import (
    ProviderIdentity,
    ProviderTurn,
    RecoveryOutcome,
    RecoveryRuntime,
    record_provider_turn,
    recover_coordinator,
)
from milknado.domains.graph import ExecutionGroup, MikadoGraph


class _RecoveryPorts:
    def __init__(self) -> None:
        self.provider_calls: list[tuple[ProviderIdentity, Path]] = []
        self.worktree_calls: list[ExecutionGroup] = []

    def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome:
        self.provider_calls.append((identity, cwd))
        return "unsupported"

    def restore(self, group: ExecutionGroup) -> bool:
        self.worktree_calls.append(group)
        return False


def _coordinator_id(graph: MikadoGraph) -> str:
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    with closing(sqlite3.connect(graph.db_path)) as conn:
        return start_coordinator(conn, goal.id, "codex").id


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        (
            ProviderBinding("invalid", "group", "codex", "provider"),
            "invalid provider binding scope",
        ),
        (
            ProviderBinding("coordinator", "wrong", "codex", "provider"),
            "wrong scope identity",
        ),
        (
            ProviderBinding("execution_group", "", "codex", "provider"),
            "identities must not be empty",
        ),
        (ProviderBinding("execution_group", "group", "codex", ""), "identities must not be empty"),
        (
            ProviderBinding("execution_group", "group", "other", "provider"),
            "unsupported provider family",
        ),
    ],
)
def test_invalid_binding_does_not_persist(
    graph: MikadoGraph, binding: ProviderBinding, message: str
) -> None:
    coordinator_id = _coordinator_id(graph)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        with pytest.raises(ValueError, match=message):
            bind_provider_session(conn, coordinator_id, binding)
        assert conn.execute("SELECT * FROM coordinator_provider_bindings").fetchall() == []
        assert conn.execute("SELECT * FROM coordinator_links").fetchall() == []
        assert conn.execute("SELECT * FROM coordinator_events").fetchall() == []


@pytest.mark.parametrize(
    ("provider_id", "turn_id", "message"),
    [
        ("bound", "", "provider turn identity must not be empty"),
        ("unbound", "turn-1", "provider turn has no coordinator binding"),
    ],
)
def test_invalid_turn_does_not_persist(
    graph: MikadoGraph, provider_id: str, turn_id: str, message: str
) -> None:
    coordinator_id = _coordinator_id(graph)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        bind_provider_session(
            conn, coordinator_id, ProviderBinding("coordinator", coordinator_id, "codex", "bound")
        )
        turn = ProviderTurn(ProviderIdentity("codex", provider_id), turn_id, "submitted")
        with pytest.raises(ValueError, match=message):
            record_provider_turn(conn, coordinator_id, turn)
        assert conn.execute("SELECT * FROM coordinator_turn_events").fetchall() == []
        assert conn.execute("SELECT * FROM coordinator_events").fetchall() == []


def test_relative_root_rejects_before_external_recovery(
    graph: MikadoGraph,
) -> None:
    coordinator_id = _coordinator_id(graph)
    ports = _RecoveryPorts()
    with closing(sqlite3.connect(graph.db_path)) as conn:
        link_entity(conn, coordinator_id, "provider_session", "bound")
        bind_provider_session(
            conn, coordinator_id, ProviderBinding("coordinator", coordinator_id, "codex", "bound")
        )
        with pytest.raises(ValueError, match="project root must be absolute"):
            _ = recover_coordinator(
                conn, coordinator_id, RecoveryRuntime(graph.groups, Path("."), ports, ports)
            )
        assert conn.execute("SELECT * FROM coordinator_turn_events").fetchall() == []
        assert conn.execute("SELECT * FROM coordinator_events").fetchall() == []
    assert ports.provider_calls == []
    assert ports.worktree_calls == []
