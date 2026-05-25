"""Analytics/plan-state pass-throughs for MikadoGraph, split off as a mixin.

These methods are pure delegations to `_persistence` free functions; relocating
them out of `graph.py` keeps the facade under its line-size ceiling while every
future graph operation gets headroom. Mixed into MikadoGraph, which supplies the
`self._conn` they delegate against.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from milknado.domains.graph._persistence import (
    get_latest_batch_plan,
    get_spec_hash,
    recent_completion_durations,
    record_batch_plan,
    record_completion_duration,
    set_dispatched_at,
    set_spec_hash,
)

if TYPE_CHECKING:
    from milknado.domains.batching import BatchPlan


class _AnalyticsFacade:
    _conn: sqlite3.Connection

    def record_batch_plan(self, plan: BatchPlan) -> int:
        return record_batch_plan(self._conn, plan)

    def get_latest_batch_plan(self) -> dict | None:
        return get_latest_batch_plan(self._conn)

    def record_completion_duration(self, node_id: int, duration_seconds: float) -> None:
        record_completion_duration(self._conn, node_id, duration_seconds)

    def recent_completion_durations(self, limit: int) -> list[float]:
        return recent_completion_durations(self._conn, limit)

    def set_dispatched_at(self, node_id: int) -> None:
        set_dispatched_at(self._conn, node_id)

    def set_spec_hash(self, spec_hash: str) -> None:
        set_spec_hash(self._conn, spec_hash)

    def get_spec_hash(self) -> str | None:
        return get_spec_hash(self._conn)
