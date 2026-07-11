"""Field vocabulary + lookup helpers for the GitHub Projects membrane.

The milknado-owned region of a Project item is two named fields — a single-select
Status and a free-text harvest — plus a fixed status→option mapping. Kept in one
module so bind (which creates the fields) and export (which writes them) agree on
the names and option set without either re-deriving them.
"""

from __future__ import annotations

from milknado.domains.common import HarvestSummary

# Distinct from the builtin Projects v2 "Status" (Todo/In Progress/Done) so the
# two never collide on a project that also uses the native workflow field.
STATUS_FIELD_NAME = "Milknado Status"
HARVEST_FIELD_NAME = "Milknado Harvest"
STATUS_OPTIONS = ["Pending", "Running", "Done", "Failed"]

_OPTION_BY_STATUS = {
    "pending": "Pending",
    "running": "Running",
    "done": "Done",
    "failed": "Failed",
}


def status_option_name(status_value: str) -> str | None:
    """Map a milknado node status value to its Status option, or None if unmapped."""
    return _OPTION_BY_STATUS.get(status_value)


def find_field(fields: list[dict], name: str) -> dict | None:
    """Return the field dict with this name, or None."""
    for field in fields:
        if field.get("name") == name:
            return field
    return None


def find_option_id(field: dict, option_name: str) -> str | None:
    """Return the option id for a named single-select option, or None."""
    for option in field.get("options", []):
        if option.get("name") == option_name:
            return option.get("id")
    return None


def format_harvest_text(summary: HarvestSummary) -> str:
    """Render a harvest summary as plain text for the Project harvest field."""
    lines = [
        f"result: {summary.status} · tasks: "
        f"{summary.tasks_done} done / {summary.tasks_failed} failed"
    ]
    if summary.result_summaries:
        lines.append("summary: " + " | ".join(summary.result_summaries))
    if summary.branch_names:
        lines.append("branches: " + ", ".join(summary.branch_names))
    return "\n".join(lines)
