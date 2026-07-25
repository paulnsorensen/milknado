"""Field vocabulary + harvest-text formatting for the GitHub membrane."""

from __future__ import annotations

from milknado.domains.github import GithubField
from milknado.domains.github._fields import (
    STATUS_OPTIONS,
    find_field,
    find_option_id,
    status_option_name,
)
from milknado.domains.reporting import HarvestSummary, format_harvest_text


def test_status_field_created_with_exactly_the_four_spec_options() -> None:
    # Acceptance 5 pins the option set literally: the create-once field carries
    # all four options, in order, no more, no fewer. The bind create-call test
    # compares against this same constant, so without this literal anchor a
    # regression that dropped/renamed an option would slip through both.
    assert STATUS_OPTIONS == ["Pending", "Running", "Done", "Failed"]


def test_every_mapped_status_option_exists_in_the_created_field() -> None:
    # Couple the status->option mapping to the create list so they can't drift:
    # a failed goal maps to "Failed", so "Failed" MUST be one of the options the
    # field is created with — else the export's find_option_id silently skips the
    # Status write and the membrane loses a state.
    for status in ("pending", "running", "done", "failed"):
        option = status_option_name(status)
        assert option is not None
        assert option in STATUS_OPTIONS


def test_status_option_name_maps_known_statuses() -> None:
    assert status_option_name("pending") == "Pending"
    assert status_option_name("running") == "Running"
    assert status_option_name("done") == "Done"
    assert status_option_name("failed") == "Failed"


def test_status_option_name_blocked_is_none() -> None:
    assert status_option_name("blocked") is None


def test_find_field_by_name() -> None:
    fields = [GithubField(id="A", name="A"), GithubField(id="F", name="Milknado Status")]
    assert find_field(fields, "Milknado Status") == GithubField(id="F", name="Milknado Status")
    assert find_field(fields, "Absent") is None


def test_find_option_id() -> None:
    field = GithubField.model_validate(
        {"id": "F", "name": "Status", "options": [{"name": "Done", "id": "opt-done"}]}
    )
    assert find_option_id(field, "Done") == "opt-done"
    assert find_option_id(field, "Missing") is None


def test_format_harvest_text_summary_line_only() -> None:
    summary = HarvestSummary("done", 2, 1, [], [])
    assert format_harvest_text(summary) == "result: done · tasks: 2 done / 1 failed"


def test_format_harvest_text_includes_summaries_and_branches() -> None:
    summary = HarvestSummary("done", 1, 0, ["shipped it", "and again"], ["claude/a", "claude/b"])
    text = format_harvest_text(summary)
    assert "summary: shipped it | and again" in text
    assert "branches: claude/a, claude/b" in text
    assert text.startswith("result: done · tasks: 1 done / 0 failed")
