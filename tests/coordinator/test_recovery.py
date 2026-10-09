from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.coordinator import ProviderBinding
from milknado.domains.coordinator.journal import control_history
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
    UnknownTurn,
    record_provider_turn,
    recover_coordinator,
)
from milknado.domains.graph import ExecutionGroup, GroupWorkspace, MikadoGraph
from milknado.domains.graph._persistence import create_tables, migrate


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


def _group(graph: MikadoGraph, workspace: Path, provider_id: str) -> ExecutionGroup:
    workspace.mkdir()
    return graph.groups.create(
        "graph-a",
        (graph.add_node("task").id,),
        GroupWorkspace(str(workspace), "group-branch", provider_id),
    )


def _bind(conn: sqlite3.Connection, coordinator_id: str, binding: ProviderBinding) -> None:
    link_entity(conn, coordinator_id, "provider_session", binding.provider_session_id)
    if binding.scope_kind == "execution_group":
        link_entity(conn, coordinator_id, "execution_group", binding.scope_id)
    bind_provider_session(conn, coordinator_id, binding)


@pytest.mark.parametrize("reverse_links", [False, True])
def test_mixed_provider_bindings_recover_once_after_group_restore(
    graph: MikadoGraph, tmp_path: Path, reverse_links: bool
) -> None:
    coordinator_id = _session(graph)
    group = _group(graph, tmp_path / "group", "claude-session")
    main = ProviderBinding("coordinator", coordinator_id, "codex", "codex-session")
    worker = ProviderBinding("execution_group", group.id, "claude", "claude-session")
    links = (
        ("provider_session", main.provider_session_id),
        ("provider_session", worker.provider_session_id),
        ("execution_group", group.id),
    )
    with closing(sqlite3.connect(graph.db_path)) as conn:
        for kind, entity_id in reversed(links) if reverse_links else links:
            link_entity(conn, coordinator_id, kind, entity_id)
        bind_provider_session(conn, coordinator_id, main)
        bind_provider_session(conn, coordinator_id, worker)
    provider = ProviderPort("resumed")
    worktrees = WorktreePort()
    with closing(sqlite3.connect(graph.db_path)) as conn:
        result = recover_coordinator(
            conn, coordinator_id, RecoveryRuntime(graph.groups, tmp_path, provider, worktrees)
        )
        statuses = [
            event.status
            for event in control_history(conn, coordinator_id)
            if event.kind == "recovery"
        ]
    assert worktrees.calls == [group]
    assert provider.calls == [
        (ProviderIdentity("codex", "codex-session"), tmp_path),
        (ProviderIdentity("claude", "claude-session"), tmp_path / "group"),
    ]
    assert [receipt.outcome for receipt in result.receipts] == ["resumed", "resumed"]
    assert statuses == ["resumed", "resumed"]


def test_restore_failure_never_resumes_group(graph: MikadoGraph, tmp_path: Path) -> None:
    coordinator_id = _session(graph)
    group = _group(graph, tmp_path / "group", "group-provider")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        _bind(
            conn,
            coordinator_id,
            ProviderBinding("execution_group", group.id, "claude", "group-provider"),
        )
    provider = ProviderPort("resumed")
    worktrees = WorktreePort(restored=False)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        result = recover_coordinator(
            conn, coordinator_id, RecoveryRuntime(graph.groups, tmp_path, provider, worktrees)
        )
    assert worktrees.calls == [group]
    assert provider.calls == []
    assert result.receipts[0].outcome == "unavailable"


def test_unbound_or_mismatched_group_fails_before_any_provider_call(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    coordinator_id = _session(graph)
    group = _group(graph, tmp_path / "group", "actual-provider")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        _bind(
            conn,
            coordinator_id,
            ProviderBinding("coordinator", coordinator_id, "codex", "main-provider"),
        )
        _bind(
            conn,
            coordinator_id,
            ProviderBinding("execution_group", group.id, "claude", "wrong-provider"),
        )
    provider = ProviderPort()
    with closing(sqlite3.connect(graph.db_path)) as conn:
        with pytest.raises(ValueError, match="identity mismatch"):
            _ = recover_coordinator(
                conn,
                coordinator_id,
                RecoveryRuntime(graph.groups, tmp_path, provider, WorktreePort()),
            )
    assert provider.calls == []


def test_turns_are_scoped_and_unknown_transition_is_idempotent(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    coordinator_id = _session(graph)
    group = _group(graph, tmp_path / "group", "two")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        _bind(conn, coordinator_id, ProviderBinding("coordinator", coordinator_id, "codex", "one"))
        _bind(conn, coordinator_id, ProviderBinding("execution_group", group.id, "claude", "two"))
        record_provider_turn(
            conn,
            coordinator_id,
            ProviderTurn(ProviderIdentity("codex", "one"), "turn-1", "submitted"),
        )
        record_provider_turn(
            conn,
            coordinator_id,
            ProviderTurn(ProviderIdentity("claude", "two"), "turn-1", "confirmed"),
        )
    runtime = RecoveryRuntime(graph.groups, tmp_path, ProviderPort("reattached"), WorktreePort())
    with closing(sqlite3.connect(graph.db_path)) as conn:
        first = recover_coordinator(conn, coordinator_id, runtime)
        second = recover_coordinator(conn, coordinator_id, runtime)
        rows = conn.execute(
            "SELECT provider_family, provider_session_id, turn_id, status "
            + "FROM coordinator_turn_events WHERE status = 'unknown'"
        ).fetchall()
        events = [
            event
            for event in control_history(conn, coordinator_id)
            if event.kind == "provider_turn"
        ]
    assert rows == [("codex", "one", "turn-1", "unknown")]
    assert [event.status for event in events] == ["submitted", "confirmed", "unknown"]
    assert (
        first.unknown_turns
        == second.unknown_turns
        == (UnknownTurn(ProviderIdentity("codex", "one"), "turn-1"),)
    )
    assert [receipt.outcome for receipt in first.receipts] == ["reattached", "reattached"]


def test_binding_uniqueness_and_invalid_turns_fail_closed(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    coordinator_id = _session(graph)
    group = _group(graph, tmp_path / "group", "same-provider")
    identity = ProviderIdentity("codex", "same-provider")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        _bind(
            conn,
            coordinator_id,
            ProviderBinding("coordinator", coordinator_id, "codex", "same-provider"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            bind_provider_session(
                conn,
                coordinator_id,
                ProviderBinding("execution_group", group.id, "codex", "same-provider"),
            )
        with pytest.raises(ValueError, match="provider evidence"):
            record_provider_turn(conn, coordinator_id, ProviderTurn(identity, "turn-1", "unknown"))
    other_coordinator = _session(graph)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            bind_provider_session(
                conn,
                other_coordinator,
                ProviderBinding("coordinator", other_coordinator, "codex", "same-provider"),
            )


def test_missing_coordinator_and_unlinked_session_fail_closed(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    provider = ProviderPort()
    with closing(sqlite3.connect(graph.db_path)) as conn:
        with pytest.raises(ValueError, match="coordinator"):
            _ = recover_coordinator(
                conn, "missing", RecoveryRuntime(graph.groups, tmp_path, provider, WorktreePort())
            )
    coordinator_id = _session(graph)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        result = recover_coordinator(
            conn, coordinator_id, RecoveryRuntime(graph.groups, tmp_path, provider, WorktreePort())
        )
    assert result.receipts == ()
    assert provider.calls == []


def test_turn_transition_and_control_event_commit_together(graph: MikadoGraph) -> None:
    coordinator_id = _session(graph)
    with closing(sqlite3.connect(graph.db_path)) as conn:
        _bind(conn, coordinator_id, ProviderBinding("coordinator", coordinator_id, "codex", "one"))
        _ = conn.execute("""
            CREATE TRIGGER reject_turn_history BEFORE INSERT ON coordinator_events
            WHEN NEW.kind = 'provider_turn'
            BEGIN SELECT RAISE(ABORT, 'history unavailable'); END
        """)
        with pytest.raises(sqlite3.IntegrityError, match="history unavailable"):
            record_provider_turn(
                conn,
                coordinator_id,
                ProviderTurn(ProviderIdentity("codex", "one"), "turn-1", "submitted"),
            )
        assert conn.execute("SELECT COUNT(*) FROM coordinator_turn_events").fetchone() == (0,)


def test_fresh_graph_migration_creates_recovery_tables(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "fresh.db")) as conn:
        create_tables(conn)
        migrate(conn)
        rows = cast(
            list[tuple[str]],
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                + "AND name IN ('coordinator_provider_bindings', 'coordinator_turn_events')"
            ).fetchall(),
        )
    assert {row[0] for row in rows} == {
        "coordinator_provider_bindings",
        "coordinator_turn_events",
    }
