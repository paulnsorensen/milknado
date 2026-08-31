"""Verification-message checks for graph status transitions."""

from __future__ import annotations

import json
import sqlite3
from typing import cast

from milknado.domains.common import MikadoNode
from milknado.domains.graph._sqlite_rows import as_tuple as _values
from milknado.domains.graph._sqlite_rows import fetchall
from milknado.domains.graph.status_flow import CLAIM_ROLE, VERIFY_ROLE


def validate_done_verification(
    conn: sqlite3.Connection, root_id: int, nodes: list[MikadoNode]
) -> None:
    rows = fetchall(
        conn,
        """
        WITH RECURSIVE subtree(id) AS (
            SELECT ?
            UNION
            SELECT e.child_id FROM edges e JOIN subtree s ON e.parent_id = s.id
        )
        SELECT n.id, r.run_id, m.role, m.body, m.seq
        FROM subtree s
        JOIN nodes n ON n.id = s.id
        JOIN runs r ON r.run_id = n.run_id
        LEFT JOIN run_messages m ON m.run_id = r.run_id
        ORDER BY m.seq
        """,
        (root_id,),
    )
    existing: set[int] = set()
    messages: dict[int, dict[str, str]] = {}
    for raw_row in rows:
        values = _values(raw_row)
        node_id = cast(int, values[0])
        role = cast(str | None, values[2])
        body = cast(str, values[3])
        existing.add(node_id)
        if role is not None:
            messages.setdefault(node_id, {})[role] = body
    for node in nodes:
        latest = messages.get(node.id, {})
        if node.id not in existing or not ({CLAIM_ROLE, VERIFY_ROLE} & latest.keys()):
            continue
        try:
            decoded = cast(object, json.loads(latest.get(VERIFY_ROLE, "")))
            verified = (
                isinstance(decoded, dict) and cast(dict[str, object], decoded).get("ok") is True
            )
        except (ValueError, TypeError, AttributeError):
            verified = False
        if not verified:
            raise ValueError(
                f"node {node.id} cannot be marked done: milknado_node_verify(run_id="
                + f"{node.run_id!r}) has not returned ok=True. Run milknado_node_verify first."
            )
