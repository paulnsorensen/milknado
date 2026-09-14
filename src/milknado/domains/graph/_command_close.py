"""Atomic revocation of a session owner and its unclaimed commands."""

import sqlite3

from milknado.domains.graph._command_records import record_receipt, utc_iso
from milknado.domains.graph._sqlite_rows import fetchall


def close_owner(
    conn: sqlite3.Connection, run_id: str, owner_incarnation: str, invocation_id: str
) -> None:
    timestamp = utc_iso(None)
    fence = (run_id, owner_incarnation, invocation_id)
    detail = "session owner closed before submission"
    _ = conn.execute("BEGIN IMMEDIATE")
    with conn:
        _ = conn.execute(
            """UPDATE owner_capabilities
               SET actions_json = '[]', permission_ids_json = '[]', published_at = ?
               WHERE run_id = ? AND owner_incarnation = ? AND invocation_id = ?""",
            (timestamp, *fence),
        )
        rows = fetchall(
            conn,
            """SELECT command_id FROM session_commands
               WHERE run_id = ? AND owner_incarnation = ? AND invocation_id = ?
                 AND status = 'queued'""",
            fence,
        )
        _ = conn.execute(
            """UPDATE session_commands
               SET status = 'rejected', updated_at = ?, detail = ?
               WHERE run_id = ? AND owner_incarnation = ? AND invocation_id = ?
                 AND status = 'queued'""",
            (timestamp, detail, *fence),
        )
        for row in rows:
            record_receipt(conn, row[0], "rejected", timestamp, detail)
