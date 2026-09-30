"""Per-invocation worker lifeline; graph access is injected by its adapter."""

from __future__ import annotations

import logging
import os
import selectors
import sqlite3
import time
from typing import Protocol

from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity
from milknado.loop._process_lifecycle import (
    Descendant,
    identity_state,
    observe_descendants,
    terminate_verified,
)


_log = logging.getLogger(__name__)

class WorkerEvidence(Protocol):
    invocation_id: str
    pid: int
    pgid: int
    start_token: float
    helper_pid: int | None
    helper_start_token: float | None
    helper_generation: int
    snapshot_seq: int
    observation_owner: str | None
    descendants: tuple[Descendant, ...]
    ended_at: str | None


class EvidenceStore(Protocol):
    def set_deadline(self, deadline: float) -> None: ...
    def get(self, invocation_id: str) -> WorkerEvidence | None: ...
    def begin(self, key: ObservationKey) -> None: ...
    def commit(self, key: ObservationKey, descendants: tuple[Descendant, ...]) -> None: ...
    def ready(self, helper: HelperIdentity, sequence: int) -> bool: ...
    def end(self, invocation_id: str) -> None: ...


def _worker(record: WorkerEvidence) -> WorkerIdentity:
    return WorkerIdentity(
        record.invocation_id, record.pid, record.pgid, record.start_token
    )


def _current(record: WorkerEvidence | None, helper: HelperIdentity) -> bool:
    return (
        record is not None
        and record.ended_at is None
        and record.helper_generation == helper.generation
        and record.helper_pid == helper.pid
        and record.helper_start_token == helper.start_token
    )


def _await_record(
    store: EvidenceStore, helper: HelperIdentity, deadline: float
) -> WorkerEvidence | None:
    while time.monotonic() < deadline:
        record = store.get(helper.invocation_id)
        if _current(record, helper):
            return record
        time.sleep(0.02)
    return None


def _cleanup(store: EvidenceStore, helper: HelperIdentity, deadline: float) -> int:
    store.set_deadline(deadline)
    try:
        record = store.get(helper.invocation_id)
    except (sqlite3.OperationalError, TimeoutError) as exc:
        _log.warning("lifeline evidence unavailable: %s", exc)
        return 1
    if not _current(record, helper):
        return 1
    assert record is not None
    worker = _worker(record)
    if record.observation_owner is None:
        key = ObservationKey(
            helper.invocation_id, "helper", record.snapshot_seq + 1,
            helper.generation, helper.pid, helper.start_token,
        )
        try:
            store.begin(key)
            observed = observe_descendants(worker)
            store.commit(key, observed)
        except (OSError, RuntimeError, sqlite3.OperationalError, TimeoutError) as exc:
            _log.warning("lifeline observation unresolved invocation=%s: %s", helper.invocation_id, exc)
            return 1
    try:
        current = store.get(helper.invocation_id)
    except (sqlite3.OperationalError, TimeoutError) as exc:
        _log.warning("lifeline evidence unavailable: %s", exc)
        return 1
    if not _current(current, helper):
        return 1
    assert current is not None
    result = terminate_verified(worker, current.descendants, deadline)
    if not result.covered_exited or current.observation_owner is not None:
        return 1
    try:
        store.end(helper.invocation_id)
    except (sqlite3.OperationalError, TimeoutError) as exc:
        _log.warning("lifeline completion unresolved: %s", exc)
        return 1
    return 0


def run_lifeline(read_fd: int, helper: HelperIdentity, store: EvidenceStore) -> int:
    """Arm EOF before READY; on parent loss persist discovery before signaling."""
    with selectors.DefaultSelector() as selector:
        selector.register(read_fd, selectors.EVENT_READ)
        record = _await_record(store, helper, time.monotonic() + 8)
        if record is None:
            return 1
        if identity_state(record.pid, record.start_token) != "live":
            return 1
        if not store.ready(helper, record.snapshot_seq):
            return 1
        print(
            f"READY {helper.invocation_id} {helper.generation} {helper.pid} "
            f"{helper.start_token} {record.snapshot_seq}",
            flush=True,
        )
        while True:
            if selector.select(timeout=1):
                if os.read(read_fd, 1) == b"":
                    return _cleanup(store, helper, time.monotonic() + 3)
            record = store.get(helper.invocation_id)
            if not _current(record, helper):
                return 1
