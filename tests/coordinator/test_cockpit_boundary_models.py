from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from milknado.domains.coordinator.model import CoordinatorSessionSummary
from milknado.domains.coordinator.plans import PlanProposalRecord
from milknado.domains.planning import decode_manifest


def test_session_summary_is_frozen_json_boundary() -> None:
    summary = CoordinatorSessionSummary("session", 7, "codex", "2026-10-08", "Deliver")

    assert isinstance(summary, msgspec.Struct)
    assert msgspec.json.decode(msgspec.json.encode(summary)) == {
        "id": "session",
        "goal_id": 7,
        "provider": "codex",
        "created_at": "2026-10-08",
        "description": "Deliver",
    }
    field = "description"
    with pytest.raises(AttributeError):
        setattr(summary, field, "Changed")


def test_plan_proposal_record_preserves_nested_json_and_proposal() -> None:
    manifest = decode_manifest(
        {
            "manifest_version": "milknado.plan.v2",
            "goal": "Deliver",
            "goal_summary": "Deliver",
            "changes": [{"id": "task-1", "path": "src/a.py", "description": "Implement"}],
            "new_relationships": [],
        }
    )
    record = PlanProposalRecord("proposal", "session", manifest, "/tmp/context.md", 3, "pending")

    assert isinstance(record, msgspec.Struct)
    assert msgspec.json.decode(msgspec.json.encode(record)) == {
        "id": "proposal",
        "session_id": "session",
        "manifest": msgspec.json.decode(msgspec.json.encode(manifest)),
        "context_path": "/tmp/context.md",
        "graph_revision": 3,
        "status": "pending",
    }
    assert record.proposal().manifest == manifest
    assert record.proposal().context_path == Path("/tmp/context.md")
    field = "status"
    with pytest.raises(AttributeError):
        setattr(record, field, "applied")
