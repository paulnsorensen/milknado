from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import msgspec

from milknado.domains.graph import graph_revision
from milknado.domains.planning import (
    PlanChangeManifest,
    PlanProposal,
    decode_manifest,
    manifest_to_dict,
)


class PlanProposalRecord(msgspec.Struct, frozen=True):
    id: str
    session_id: str
    manifest: PlanChangeManifest
    context_path: str
    graph_revision: int
    status: str

    def proposal(self) -> PlanProposal:
        return PlanProposal(self.manifest, Path(self.context_path))


def _record(row: tuple[str, str, str, str, int, str]) -> PlanProposalRecord:
    return PlanProposalRecord(
        row[0],
        row[1],
        decode_manifest(msgspec.json.decode(row[2].encode(), type=object)),
        row[3],
        row[4],
        row[5],
    )


def get_proposal(conn: sqlite3.Connection, proposal_id: str) -> PlanProposalRecord:
    row = cast(
        tuple[str, str, str, str, int, str] | None,
        conn.execute(
            "SELECT id, session_id, manifest_json, context_path, graph_revision, status "
            + "FROM coordinator_plan_proposals WHERE id = ?",
            (proposal_id,),
        ).fetchone(),
    )
    if row is None:
        raise KeyError(proposal_id)
    return _record(row)


def list_proposals(conn: sqlite3.Connection, session_id: str) -> tuple[PlanProposalRecord, ...]:
    rows = cast(
        list[tuple[str, str, str, str, int, str]],
        conn.execute(
            "SELECT id, session_id, manifest_json, context_path, graph_revision, status "
            + "FROM coordinator_plan_proposals WHERE session_id = ? ORDER BY rowid",
            (session_id,),
        ).fetchall(),
    )
    return tuple(_record(row) for row in rows)


def save_proposal(conn: sqlite3.Connection, record: PlanProposalRecord) -> PlanProposalRecord:
    with conn:
        _ = conn.execute("BEGIN IMMEDIATE")
        if graph_revision(conn) != record.graph_revision:
            raise ValueError("graph changed during planning; request a new proposal")
        _ = conn.execute(
            "INSERT INTO coordinator_plan_proposals "
            + "(id, session_id, manifest_json, context_path, graph_revision, status) "
            + "VALUES (?, ?, ?, ?, ?, 'pending')",
            (
                record.id,
                record.session_id,
                json.dumps(manifest_to_dict(record.manifest)),
                record.context_path,
                record.graph_revision,
            ),
        )
    return get_proposal(conn, record.id)


def transition_proposal(
    conn: sqlite3.Connection, proposal_id: str, expected: str, status: str
) -> PlanProposalRecord:
    with conn:
        cursor = conn.execute(
            "UPDATE coordinator_plan_proposals SET status = ? WHERE id = ? AND status = ?",
            (status, proposal_id, expected),
        )
    if cursor.rowcount != 1:
        raise ValueError("plan proposal state changed")
    return get_proposal(conn, proposal_id)
