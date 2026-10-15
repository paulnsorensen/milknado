"""Fence checks at tmux proof and review-round boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from milknado.domains.common import NodeStatus
from milknado.domains.dispatch.tmux_run import read_exec_evidence
from milknado.domains.graph import MikadoGraph, StartTaskAttempt
from milknado.domains.graph._task_attempts import record_start_evidence

_NOW = "2026-01-01T00:00:00+00:00"


def _graph(path: Path) -> tuple[MikadoGraph, int]:
    graph = MikadoGraph(path)
    node = graph.add_node("task")
    assert graph.claim_node(node.id, "owner", now=_NOW)
    _ = graph.task_attempts.start(StartTaskAttempt(node.id, "owner", "owner", "owner", 2))
    graph.runs.start("owner", node.id, "run.log", _NOW, 1, attempt_id="owner")
    return graph, node.id


def test_review_rotation_rejects_missing_identity_and_stale_owner(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path / "graph.db")
    with pytest.raises(ValueError, match="must be nonempty"):
        _ = graph.task_attempts.rotate_review_round("owner", "owner", "", "new")
    assert not graph.task_attempts.rotate_review_round("owner", "stale", "new", "new")
    node = graph.get_node(node_id)
    assert node is not None and node.run_id == "owner"
    graph.close()


def test_review_rotation_rolls_back_node_when_attempt_is_final(tmp_path: Path) -> None:
    graph, node_id = _graph(tmp_path / "graph.db")
    graph.runs.start("new", node_id, "new.log", _NOW, None, attempt_id="owner")
    assert graph.task_attempts.record_nonfailure("owner", "owner", "cancelled")
    assert not graph.task_attempts.rotate_review_round("owner", "owner", "new", "new")
    node = graph.get_node(node_id)
    assert node is not None and node.status is NodeStatus.RUNNING and node.run_id == "owner"
    attempt = graph.task_attempts.get("owner")
    assert attempt is not None and attempt.outcome == "cancelled"
    graph.close()


@pytest.mark.parametrize("identity", ["missing", "wrong_node", "wrong_attempt"])
def test_review_rotation_requires_matching_linked_new_run(tmp_path: Path, identity: str) -> None:
    path = tmp_path / "graph.db"
    graph, node_id = _graph(path)
    if identity == "wrong_node":
        other = graph.add_node("other")
        graph.runs.start("new", other.id, "new.log", _NOW, None, attempt_id="owner")
    elif identity == "wrong_attempt":
        _ = graph.task_attempts.start(StartTaskAttempt(node_id, "other", "owner", "owner"))
        graph.runs.start("new", node_id, "new.log", _NOW, None, attempt_id="other")
    assert not graph.task_attempts.rotate_review_round("owner", "owner", "new", "new")
    graph.close()
    reopened = MikadoGraph(path)
    node = reopened.get_node(node_id)
    assert node is not None and node.run_id == "owner"
    attempt = reopened.task_attempts.get("owner")
    assert attempt is not None
    assert (attempt.owner_run_id, attempt.active_invocation_id, attempt.start_state) == (
        "owner",
        "owner",
        "unknown",
    )
    reopened.close()


def test_invalid_start_evidence_does_not_change_record(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path / "graph.db")
    with pytest.raises(ValueError, match="invalid task start evidence"):
        _ = record_start_evidence(graph.group_connection, ("owner", "owner", "owner"), "pane")
    attempt = graph.task_attempts.get("owner")
    assert attempt is not None and attempt.start_state == "unknown"
    graph.close()


def test_tmux_receipt_rejects_nonobject_and_bad_pid(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    _ = receipt.write_text(json.dumps(["not", "an", "object"]))
    assert read_exec_evidence(receipt, "nonce") is None
    _ = receipt.write_text(json.dumps({"nonce": "nonce", "state": "EXECUTED", "pid": True}))
    assert read_exec_evidence(receipt, "nonce") is None
    _ = receipt.write_text(
        json.dumps(
            {
                "nonce": "nonce",
                "state": "EXECUTED",
                "pid": os.getpid() + 99999999,
                "start_ticks": "1",
            }
        )
    )
    assert read_exec_evidence(receipt, "nonce") is None
