from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.coordinator.journal import append_control_event, control_history
from milknado.domains.coordinator.model import ControlEvent
from milknado.domains.coordinator.persistence import (
    create_coordinator_tables,
    link_entity,
    start_coordinator,
)
from milknado.domains.coordinator.recovery import (
    ProviderIdentity,
    RecoveryOutcome,
    RecoveryRuntime,
    recover_coordinator,
)
from milknado.domains.graph import ExecutionGroup, GroupWorkspace, MikadoGraph


class ProviderPort:
    def __init__(self, outcome: RecoveryOutcome = "unsupported") -> None:
        self.outcome: RecoveryOutcome = outcome
        self.calls: list[tuple[ProviderIdentity, Path]] = []

    def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome:
        self.calls.append((identity, cwd))
        return self.outcome


class WorktreePort:
    def __init__(self, restored: bool = True) -> None:
        self.restored: bool = restored
        self.calls: list[ExecutionGroup] = []

    def restore(self, group: ExecutionGroup) -> bool:
        self.calls.append(group)
        return self.restored and Path(group.worktree_path).is_dir()


def _session(graph: MikadoGraph, provider: str = "codex") -> str:
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    with closing(sqlite3.connect(graph.db_path)) as conn:
        return start_coordinator(conn, goal.id, provider).id


def test_reopen_restores_bound_group_and_emits_recovery_receipts(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    session_id = _session(graph)
    workspace = tmp_path / "group"
    workspace.mkdir()
    task = graph.add_node("task")
    group = graph.groups.create(
        "graph-a", (task.id,), GroupWorkspace(str(workspace), "group-branch", "group-provider")
    )
    with closing(sqlite3.connect(graph.db_path)) as conn:
        link_entity(conn, session_id, "provider_session", "coordinator-provider")
        link_entity(conn, session_id, "execution_group", group.id)
    provider = ProviderPort("resumed")
    worktrees = WorktreePort()
    with closing(sqlite3.connect(graph.db_path)) as conn:
        result = recover_coordinator(
            conn, session_id, RecoveryRuntime(graph.groups, tmp_path, provider, worktrees)
        )
        assert [receipt.outcome for receipt in result.receipts] == ["resumed", "resumed"]
        assert [receipt.identity.session_id for receipt in result.receipts] == [
            "coordinator-provider",
            "group-provider",
        ]
        statuses = [
            event.status for event in control_history(conn, session_id) if event.kind == "recovery"
        ]
        assert statuses == ["resumed", "resumed"]
    assert worktrees.calls == [group]
    assert provider.calls == [
        (ProviderIdentity("codex", "coordinator-provider"), tmp_path),
        (ProviderIdentity("codex", "group-provider"), workspace),
    ]


def test_unconfirmed_turn_is_unknown_and_never_replayed(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    session_id = _session(graph)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        link_entity(conn, session_id, "provider_session", "provider-1")
        _ = append_control_event(
            conn,
            session_id,
            ControlEvent(kind="provider_turn", entity_id="turn-1", status="submitted"),
        )
    provider = ProviderPort("reattached")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        result = recover_coordinator(
            conn, session_id, RecoveryRuntime(graph.groups, tmp_path, provider, WorktreePort())
        )
        assert result.unknown_turn_ids == ("turn-1",)
        assert result.receipts[0].outcome == "unknown_turn"
        assert not result.receipts[0].turn_confirmed
    assert len(provider.calls) == 1


def test_group_identity_mismatch_or_missing_worktree_stops_provider_recovery(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    session_id = _session(graph)
    task = graph.add_node("task")
    group = graph.groups.create(
        "graph-a",
        (task.id,),
        GroupWorkspace(str(tmp_path / "missing"), "branch", "group-provider"),
    )
    with closing(sqlite3.connect(graph.db_path)) as conn:
        link_entity(conn, session_id, "execution_group", group.id)
    provider = ProviderPort()
    with closing(sqlite3.connect(graph.db_path)) as conn:
        result = recover_coordinator(
            conn, session_id, RecoveryRuntime(graph.groups, tmp_path, provider, WorktreePort())
        )
    assert result.receipts[0].outcome == "unavailable"
    assert provider.calls == []


def test_missing_coordinator_and_unlinked_group_fail_closed(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    provider = ProviderPort()
    with closing(sqlite3.connect(graph.db_path)) as conn:
        create_coordinator_tables(conn)
        with pytest.raises(ValueError, match="coordinator"):
            _ = recover_coordinator(
                conn, "missing", RecoveryRuntime(graph.groups, tmp_path, provider, WorktreePort())
            )
    session_id = _session(graph)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        result = recover_coordinator(
            conn, session_id, RecoveryRuntime(graph.groups, tmp_path, provider, WorktreePort())
        )
    assert result.receipts == ()
    assert provider.calls == []


def test_provider_identity_rejects_unknown_family_and_empty_session() -> None:
    with pytest.raises(ValueError, match="supported provider"):
        _ = ProviderIdentity("other", "session-1")
    with pytest.raises(ValueError, match="provider session identity"):
        _ = ProviderIdentity("codex", "")
