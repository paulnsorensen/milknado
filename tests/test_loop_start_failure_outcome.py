"""A failed task-outcome write never leaves the node claimed after a start failure."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.app._loop_start_failure import LoopStartupFailure, record_start_failure
from milknado.domains.common import NodeStatus
from milknado.domains.graph import MikadoGraph, StartTaskAttempt

_NOW = "2026-01-01T00:00:00+00:00"


def _refuse(*_args: object) -> bool:
    raise OSError("database unavailable")


def _decline(*_args: object) -> bool:
    return False


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (_decline, (RuntimeError, "task outcome fence lost")),
        (_refuse, (OSError, "database unavailable")),
    ],
)
def test_outcome_write_failure_still_releases_node_and_stays_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: object,
    expected: tuple[type[Exception], str],
) -> None:
    error, message = expected
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "r1", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "r1", "r1", "r1", 3))
    monkeypatch.setattr(graph.task_attempts, "record_nonfailure", record)
    failure = LoopStartupFailure(node.id, "r1", "r1", False, OSError("x"))

    with pytest.raises(error, match=message):
        record_start_failure(graph, failure)

    current = graph.get_node(node.id)
    assert current is not None and current.status is NodeStatus.FAILED
    graph.close()
