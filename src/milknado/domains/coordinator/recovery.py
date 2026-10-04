from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from milknado.domains.coordinator.journal import append_control_event, control_history
from milknado.domains.coordinator.model import ControlEvent, CoordinatorSession
from milknado.domains.coordinator.persistence import (
    get_coordinator,
    link_entity,
    links_for_session,
)
from milknado.domains.graph import ExecutionGroup, ExecutionGroupStore

RecoveryOutcome = Literal["reattached", "resumed", "unknown_turn", "unavailable", "unsupported"]
_OUTCOMES = frozenset({"reattached", "resumed", "unknown_turn", "unavailable", "unsupported"})


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    family: str
    session_id: str

    def __post_init__(self) -> None:
        if self.family not in {"claude", "codex"} or not self.session_id:
            raise ValueError("recovery requires a supported provider session identity")


@dataclass(frozen=True, slots=True)
class RecoveryReceipt:
    entity_kind: str
    entity_id: str
    identity: ProviderIdentity
    worktree_path: Path
    outcome: RecoveryOutcome
    turn_confirmed: bool = False


@dataclass(frozen=True, slots=True)
class CoordinatorRecovery:
    session: CoordinatorSession
    receipts: tuple[RecoveryReceipt, ...]
    unknown_turn_ids: tuple[str, ...]  # noqa: V107


class ProviderRecoveryPort(Protocol):
    def recover(self, identity: ProviderIdentity, cwd: Path) -> RecoveryOutcome: ...


class WorktreeRecoveryPort(Protocol):
    def restore(self, group: ExecutionGroup) -> bool: ...


@dataclass(frozen=True, slots=True)
class RecoveryRuntime:
    groups: ExecutionGroupStore
    root: Path
    provider: ProviderRecoveryPort
    worktrees: WorktreeRecoveryPort


def _unknown_turns(conn: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    states: dict[str, str] = {}
    for event in control_history(conn, session_id):
        if event.kind == "provider_turn" and event.entity_id:
            states[event.entity_id] = event.status
    return tuple(turn_id for turn_id, status in states.items() if status != "confirmed")


def _record_receipt(
    conn: sqlite3.Connection, session_id: str, receipt: RecoveryReceipt
) -> RecoveryReceipt:
    if receipt.outcome not in _OUTCOMES:
        raise ValueError(f"invalid recovery outcome: {receipt.outcome}")
    seq = append_control_event(
        conn,
        session_id,
        ControlEvent(
            kind="recovery",
            text="provider recovery result",
            entity_kind=receipt.entity_kind,
            entity_id=receipt.entity_id,
            status=receipt.outcome,
        ),
    )
    link_entity(conn, session_id, "recovery", str(seq))
    return receipt


def recover_coordinator(
    conn: sqlite3.Connection, session_id: str, runtime: RecoveryRuntime
) -> CoordinatorRecovery:
    """Restore linked identities without replaying an unconfirmed provider turn."""
    session = get_coordinator(conn, session_id)
    if session is None:
        raise ValueError("coordinator session does not exist")
    if not runtime.root.is_absolute():
        raise ValueError("project root must be absolute")
    links = links_for_session(conn, session_id)
    unknown = _unknown_turns(conn, session_id)
    receipts: list[RecoveryReceipt] = []
    for link in links:
        if link.kind == "provider_session":
            identity = ProviderIdentity(session.provider, link.entity_id)
            outcome = runtime.provider.recover(identity, runtime.root)
            if unknown and outcome in {"reattached", "resumed"}:
                outcome = "unknown_turn"
            receipts.append(
                _record_receipt(
                    conn,
                    session_id,
                    RecoveryReceipt(
                        "provider_session", link.entity_id, identity, runtime.root, outcome
                    ),
                )
            )
        elif link.kind == "execution_group":
            group = runtime.groups.get(link.entity_id)
            if group is None:
                raise ValueError(f"linked execution group does not exist: {link.entity_id}")
            identity = ProviderIdentity(session.provider, group.provider_session_id)
            path = Path(group.worktree_path)
            if (
                not group.provider_session_id
                or not path.is_absolute()
                or not group.branch_name
                or not runtime.worktrees.restore(group)
            ):
                outcome = "unavailable"
            else:
                outcome = runtime.provider.recover(identity, path)
            if unknown and outcome in {"reattached", "resumed"}:
                outcome = "unknown_turn"
            receipts.append(
                _record_receipt(
                    conn,
                    session_id,
                    RecoveryReceipt("execution_group", group.id, identity, path, outcome),
                )
            )
    return CoordinatorRecovery(session, tuple(receipts), unknown)
