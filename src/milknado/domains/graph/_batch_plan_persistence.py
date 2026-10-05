from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from milknado.domains.batching import BatchPlan


def record_batch_plan(conn: sqlite3.Connection, plan: BatchPlan) -> int:
    owns_transaction = not conn.in_transaction
    spread_payload = [
        {"symbol_name": item.symbol.name, "symbol_file": item.symbol.file, "spread": item.spread}
        for item in plan.spread_report
    ]
    max_spread = max((item.spread for item in plan.spread_report), default=0)
    oversized_count = sum(1 for b in plan.batches if b.oversized)
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        "INSERT INTO batch_plans "
        + "(created_at, solver_status, batch_count, oversized_count, max_spread, spread_json) "
        + "VALUES (?, ?, ?, ?, ?, ?)",
        (
            now,
            plan.solver_status,
            len(plan.batches),
            oversized_count,
            max_spread,
            json.dumps(spread_payload),
        ),
    )
    if owns_transaction:
        conn.commit()
    plan_id = cur.lastrowid
    if plan_id is None:  # pragma: no cover - defensive: plain INSERT always sets lastrowid
        raise RuntimeError("record_batch_plan INSERT did not return lastrowid")
    return plan_id
