from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest

from milknado.domains.coordinator import CoordinatorControl, ProviderBinding
from milknado.domains.coordinator.control_models import (
    CoordinatorCommandReceipt,
    Recover,
    StartGoal,
)
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.coordinator.journal import control_history
from milknado.domains.coordinator.model import RecoveryReceipt as ModelRecoveryReceipt
from milknado.domains.coordinator.persistence import bind_provider_session, link_entity
from milknado.domains.coordinator.recovery import (
    ProviderIdentity,
    ProviderTurn,
    RecoveryOutcome,
    RecoveryReceipt,
    RecoveryRuntime,
    record_provider_turn,
)
from milknado.domains.coordinator.recovery_receipts import record_recovery_receipt
from milknado.domains.graph import ExecutionGroup, GroupWorkspace, MikadoGraph


class _BlockingProvider:
    def __init__(self, entered: Event, release: Event) -> None:
        self.entered: Event = entered
        self.release: Event = release

    def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome:
        _ = identity, cwd
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("provider probe did not resume")
        return "resumed"


class _Worktrees:
    def restore(self, group: ExecutionGroup) -> bool:
        _ = group
        return True


def test_provider_probe_does_not_hold_graph_lock(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    entered, release, acquired = Event(), Event(), Event()
    runtime = RecoveryRuntime(
        graph.groups, tmp_path, _BlockingProvider(entered, release), _Worktrees()
    )
    control = CoordinatorControl(graph, tmp_path, CoordinatorServices(recovery_runtime=runtime))
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    with graph.synchronization_lock:
        bind_provider_session(
            graph.group_connection,
            session_id,
            ProviderBinding("coordinator", session_id, "codex", "provider"),
        )
        link_entity(graph.group_connection, session_id, "provider_session", "provider")

    receipts: list[CoordinatorCommandReceipt] = []
    worker = Thread(
        target=lambda: receipts.append(
            control.send_coordinator_command(session_id, Recover("recover"))
        )
    )
    contender = Thread(target=lambda: _acquire_graph_lock(graph, acquired))
    try:
        worker.start()
        assert entered.wait(5)
        contender.start()
        assert acquired.wait(1), "provider probe held the graph lock"
    finally:
        release.set()
        worker.join(5)
        if contender.ident is not None:
            contender.join(5)
        graph.close()
    assert len(receipts) == 1
    assert receipts[0].status == "accepted"


def _acquire_graph_lock(graph: MikadoGraph, acquired: Event) -> None:
    with graph.synchronization_lock:
        acquired.set()


@pytest.fixture
def bound_recovery(
    tmp_path: Path,
) -> tuple[MikadoGraph, CoordinatorControl, str, str, Event, Event]:
    graph = MikadoGraph(tmp_path / "graph.db")
    entered, release = Event(), Event()
    runtime = RecoveryRuntime(
        graph.groups, tmp_path, _BlockingProvider(entered, release), _Worktrees()
    )
    control = CoordinatorControl(graph, tmp_path, CoordinatorServices(recovery_runtime=runtime))
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    worktree = tmp_path / "group"
    worktree.mkdir()
    group = graph.groups.create(
        "graph", (graph.add_node("task").id,), GroupWorkspace(str(worktree), "branch", None)
    )
    with graph.synchronization_lock:
        bind_provider_session(
            graph.group_connection,
            session_id,
            ProviderBinding("coordinator", session_id, "codex", "provider"),
        )
        link_entity(graph.group_connection, session_id, "provider_session", "provider")
        link_entity(graph.group_connection, session_id, "execution_group", group.id)
    return graph, control, session_id, group.id, entered, release


@pytest.mark.parametrize("mutation", ["binding", "group", "link", "session", "turn"])
def test_concurrent_owner_or_turn_change_rejects_recovery(
    bound_recovery: tuple[MikadoGraph, CoordinatorControl, str, str, Event, Event], mutation: str
) -> None:
    graph, control, session_id, group_id, entered, release = bound_recovery
    receipts: list[CoordinatorCommandReceipt] = []
    worker = Thread(
        target=lambda: receipts.append(
            control.send_coordinator_command(session_id, Recover("recover"))
        )
    )
    try:
        worker.start()
        assert entered.wait(5)
        with graph.synchronization_lock:
            _change_recovery_input(graph, session_id, group_id, mutation)
    finally:
        release.set()
        worker.join(5)
    assert len(receipts) == 1
    assert receipts[0].status == "rejected"
    assert "changed during probe" in cast(str, receipts[0].result)
    assert not any(
        event.kind == "recovery" for event in control_history(graph.group_connection, session_id)
    )
    if mutation == "turn":
        rows = graph.group_connection.execute(
            "SELECT status FROM coordinator_turn_events WHERE coordinator_id = ? AND turn_id = ?",
            (session_id, "new-turn"),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "submitted"
    graph.close()


def test_recovery_receipt_has_concrete_owner_and_preserves_identity(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    started = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    session_id = cast(str, cast(dict[str, object], started.result)["id"])
    assert RecoveryReceipt is ModelRecoveryReceipt
    receipt = RecoveryReceipt(
        "coordinator", session_id, ProviderIdentity("codex", "provider"), tmp_path, "resumed"
    )
    assert record_recovery_receipt(graph.group_connection, session_id, receipt) is receipt
    assert any(
        event.kind == "recovery" for event in control_history(graph.group_connection, session_id)
    )
    graph.close()


def _change_recovery_input(
    graph: MikadoGraph, session_id: str, group_id: str, mutation: str
) -> None:
    if mutation == "turn":
        record_provider_turn(
            graph.group_connection,
            session_id,
            ProviderTurn(ProviderIdentity("codex", "provider"), "new-turn", "submitted"),
        )
        return
    statements = {
        "binding": (
            "DELETE FROM coordinator_provider_bindings WHERE coordinator_id = ?",
            session_id,
        ),
        "group": (
            "UPDATE execution_groups SET branch_name = 'new-branch' WHERE id = ?",
            group_id,
        ),
        "link": (
            "DELETE FROM coordinator_links WHERE session_id = ? AND kind = 'provider_session'",
            session_id,
        ),
        "session": (
            "UPDATE coordinator_sessions SET provider = 'claude' WHERE id = ?",
            session_id,
        ),
    }
    statement, identity = statements[mutation]
    with graph.group_connection:
        _ = graph.group_connection.execute(statement, (identity,))
