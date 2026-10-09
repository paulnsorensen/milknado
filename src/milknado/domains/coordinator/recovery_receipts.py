"""Persist recovery receipts in the coordinator control journal."""

from __future__ import annotations

import sqlite3

from milknado.domains.coordinator.journal import append_control_event
from milknado.domains.coordinator.model import ControlEvent, RecoveryReceipt
from milknado.domains.coordinator.persistence import link_entity

_OUTCOMES = frozenset({"reattached", "resumed", "unknown_turn", "unavailable", "unsupported"})


def record_recovery_receipt(
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
