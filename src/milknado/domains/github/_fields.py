"""Field vocabulary shared by GitHub bind and export."""

from __future__ import annotations

from collections.abc import Sequence

from milknado.domains.github.models import GithubField

STATUS_FIELD_NAME = "Milknado Status"
HARVEST_FIELD_NAME = "Milknado Harvest"
STATUS_OPTIONS = ["Pending", "Running", "Done", "Blocked", "Failed"]

_OPTION_BY_STATUS = {
    "pending": "Pending",
    "running": "Running",
    "done": "Done",
    "blocked": "Blocked",
    "failed": "Failed",
}


def status_option_name(status_value: str) -> str | None:
    return _OPTION_BY_STATUS.get(status_value)


def find_field(fields: Sequence[GithubField], name: str) -> GithubField | None:
    return next((field for field in fields if field.name == name), None)


def find_option_id(field: GithubField, option_name: str) -> str | None:
    option = next((option for option in field.options if option.name == option_name), None)
    return option.id if option else None
