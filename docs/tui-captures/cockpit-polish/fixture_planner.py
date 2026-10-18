"""Deterministic capture planner: prints one fixed milknado.plan.v2 manifest.

Used through `planning_agent = "local-planner"` and `MILKNADO_LOCAL_PLANNER`.
"""

import json
import sys

_ = sys.stdin.read()
manifest = {
    "manifest_version": "milknado.plan.v2",
    "goal": "Ship the billing dashboard",
    "goal_summary": (
        "Add the billing dashboard page and its usage chart. "
        "Success means the page renders current-month usage from the existing API."
    ),
    "spec_path": None,
    "changes": [
        {
            "id": "c1",
            "path": "web/billing/page.tsx",
            "edit_kind": "add",
            "description": "Add the billing dashboard page shell.",
            "depends_on": [],
        },
        {
            "id": "c2",
            "path": "web/billing/usage-chart.tsx",
            "edit_kind": "add",
            "description": "Render current-month usage as a bar chart.",
            "depends_on": ["c1"],
        },
    ],
    "new_relationships": [],
}
print("```json\n" + json.dumps(manifest, indent=2) + "\n```")
