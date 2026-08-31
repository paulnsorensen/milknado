"""Analytics/plan-state pass-throughs for MikadoGraph, split off as a mixin.

These methods are pure delegations to `_persistence` free functions; relocating
them out of `graph.py` keeps the facade under its line-size ceiling while every
future graph operation gets headroom. Mixed into MikadoGraph, which supplies the
`self._conn` they delegate against.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

from milknado.domains.graph._persistence import (
    BatchPlanRecord,
    get_latest_batch_plan,
    get_spec_hash,
    recent_completion_durations,
    record_batch_plan,
    record_completion_duration,
    set_dispatched_at,
    set_spec_hash,
)

__all__ = ["_AnalyticsFacade", "synchronized"]

if TYPE_CHECKING:
    from milknado.domains.batching import BatchPlan


_P = ParamSpec("_P")
_R = TypeVar("_R")


class _Lockable(Protocol):
    synchronization_lock: AbstractContextManager[object]


def synchronized(method: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(method)
    def locked(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        graph = cast(_Lockable, args[0])
        with graph.synchronization_lock:
            return method(*args, **kwargs)

    return locked


class _AnalyticsFacade:
    @property
    def _conn(self) -> sqlite3.Connection:
        raise NotImplementedError

    @synchronized
    def record_batch_plan(self, plan: BatchPlan) -> int:
        return record_batch_plan(self._conn, plan)

    @synchronized
    def get_latest_batch_plan(self) -> BatchPlanRecord | None:
        return get_latest_batch_plan(self._conn)

    @synchronized
    def record_completion_duration(self, node_id: int, duration_seconds: float) -> None:
        record_completion_duration(self._conn, node_id, duration_seconds)

    @synchronized
    def recent_completion_durations(self, limit: int) -> list[float]:
        return recent_completion_durations(self._conn, limit)

    @synchronized
    def set_dispatched_at(self, node_id: int) -> None:
        set_dispatched_at(self._conn, node_id)

    @synchronized
    def set_spec_hash(self, spec_hash: str) -> None:
        set_spec_hash(self._conn, spec_hash)

    @synchronized
    def get_spec_hash(self) -> str | None:
        return get_spec_hash(self._conn)
