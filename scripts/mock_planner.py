#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> None:
    body = sys.stdin.read().lower()
    is_revision = "user revision request" in body
    payload = {
        "manifest_version": "milknado.plan.v2",
        "goal": "Mock planning goal",
        "goal_summary": "Revised plan from mock planner." if is_revision else "Initial mock plan.",
        "spec_path": "spec.md",
        "changes": [
            {
                "id": "c1",
                "path": "src/mock_feature.py",
                "edit_kind": "add",
                "description": "Add initial mock feature slice.",
                "depends_on": [],
            },
            {
                "id": "c2",
                "path": "tests/test_mock_feature.py",
                "edit_kind": "add",
                "description": "Add tests for mock feature.",
                "depends_on": ["c1"],
            },
        ],
        "new_relationships": [
            {
                "source_change_id": "c1",
                "dependant_change_id": "c2",
                "reason": "new_call",
            }
        ],
    }
    print("```json")
    print(json.dumps(payload))
    print("```")


if __name__ == "__main__":
    main()
