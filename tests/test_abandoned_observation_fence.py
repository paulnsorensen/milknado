"""Store fences around the takeover of an observation abandoned by a dead supervisor."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from milknado.domains.common import ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.graph import MikadoGraph, WorkerEvidenceStore, WorkerRecord

_SUPERVISOR_PID = 2**31 - 1
_SUPERVISOR_TOKEN = 123.5
_WORKER_PID = 2**31 - 2
_WORKER_TOKEN = 456.5
_COMMITTED = ((101, 1.5, 101),)


def _supervisor_key(sequence: int) -> ObservationKey:
    return ObservationKey("inv-1", "supervisor", sequence, 0, _SUPERVISOR_PID, _SUPERVISOR_TOKEN)


def _reaper_key(sequence: int) -> ObservationKey:
    return ObservationKey("inv-1", "supervisor", sequence, -1, _WORKER_PID, _WORKER_TOKEN)


@pytest.fixture
def abandoned(tmp_path: Path) -> Iterator[WorkerEvidenceStore]:
    """Commit observation 1, then leave observation 2 begun as a killed supervisor would."""
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        node = graph.add_node("recover")
        assert graph.claim_node(node.id, "run-1", now="2026-01-01T00:00:00+00:00", pid=1)
        graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        graph.runs.record_worker(
            WorkerOwner("run-1", _SUPERVISOR_PID, _SUPERVISOR_TOKEN, "run-1", node.id),
            WorkerIdentity("inv-1", _WORKER_PID, _WORKER_PID, _WORKER_TOKEN),
        )
        with WorkerEvidenceStore(graph.db_path) as store:
            store.begin(_supervisor_key(1))
            store.commit(_supervisor_key(1), _COMMITTED)
            store.begin(_supervisor_key(2))
            yield store
    finally:
        graph.close()


def _record(store: WorkerEvidenceStore) -> WorkerRecord:
    record = store.get("inv-1")
    assert record is not None
    return record


def _supervisor_moved_on(store: WorkerEvidenceStore) -> ObservationKey:
    store.commit(_supervisor_key(2), ((202, 2.5, 202),))
    store.begin(_supervisor_key(3))
    return _supervisor_key(3)


def _another_reaper_took_over(store: WorkerEvidenceStore) -> ObservationKey:
    store.release_abandoned(_record(store))
    store.begin(_reaper_key(2))
    return _reaper_key(2)


@pytest.mark.parametrize("advance", [_supervisor_moved_on, _another_reaper_took_over])
def test_stale_release_record_keeps_current_fence(
    abandoned: WorkerEvidenceStore,
    advance: Callable[[WorkerEvidenceStore], ObservationKey],
) -> None:
    stale = _record(abandoned)
    assert (stale.observation_owner, stale.observation_seq) == ("supervisor", 2)
    holder = advance(abandoned)
    before = _record(abandoned)

    with pytest.raises(RuntimeError, match="worker observation fence lost"):
        abandoned.release_abandoned(stale)

    assert _record(abandoned) == before
    assert before.observation_owner == "supervisor"
    assert before.observation_seq == holder.sequence
    abandoned.commit(holder, ())
    assert _record(abandoned).snapshot_seq == holder.sequence


def test_abandoned_supervisor_commit_after_takeover_is_rejected(
    abandoned: WorkerEvidenceStore,
) -> None:
    abandoned.release_abandoned(_record(abandoned))
    abandoned.begin(_reaper_key(2))
    taken_over = _record(abandoned)
    assert (taken_over.observation_owner, taken_over.observation_seq) == ("supervisor", 2)

    with pytest.raises(RuntimeError, match="worker observation fence lost"):
        abandoned.commit(_supervisor_key(2), ((999, 9.5, 999),))

    assert _record(abandoned) == taken_over
    assert taken_over.snapshot_seq == 1
    assert taken_over.descendants == _COMMITTED
    reaped = ((303, 3.5, 303),)
    abandoned.commit(_reaper_key(2), reaped)
    assert _record(abandoned) == replace(
        taken_over,
        snapshot_seq=2,
        observation_owner=None,
        observation_seq=None,
        descendants=tuple(sorted(_COMMITTED + reaped)),
    )


def test_abandoned_supervisor_commit_after_reaper_commit_is_rejected(
    abandoned: WorkerEvidenceStore,
) -> None:
    abandoned.release_abandoned(_record(abandoned))
    abandoned.begin(_reaper_key(2))
    abandoned.commit(_reaper_key(2), ())
    committed = _record(abandoned)

    with pytest.raises(RuntimeError, match="worker observation fence lost"):
        abandoned.commit(_supervisor_key(2), ((999, 9.5, 999),))

    assert _record(abandoned) == committed
    assert committed.observation_owner is None
    assert (committed.snapshot_seq, committed.descendants) == (2, _COMMITTED)
