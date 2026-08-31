"""`adapters/gh.py` transport: each `gh project`/`gh issue` wrapper builds the
right argv and parses the JSON reply, and NO argv ever contains
`graphql`/`api` (Acceptance 6, 8).

subprocess.run is fully mocked so these tests never touch a real `gh`.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from milknado.adapters import gh
from milknado.adapters.gh import (
    gh_field_create,
    gh_field_create_text,
    gh_field_list,
    gh_issue_create,
    gh_issue_edit_body,
    gh_issue_view,
    gh_item_add,
    gh_item_edit,
    gh_item_list,
    gh_project_view,
)
from milknado.domains.github import (
    GithubField,
    GithubIssue,
    GithubItem,
    GithubProject,
)

GH_BIN = "/usr/bin/gh"


class Recorder:
    """Stand-in for subprocess.run that records argv and returns canned stdout."""

    def __init__(self, stdout: str = "") -> None:
        self.calls: list[list[str]] = []
        self.stdout = stdout

    def __call__(self, args, **_kwargs):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=self.stdout, stderr="")


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch):
    """Pin `gh` on PATH and swap subprocess.run for a Recorder factory."""
    monkeypatch.setattr(gh.shutil, "which", lambda _name: GH_BIN)

    def _install(stdout: str = "") -> Recorder:
        rec = Recorder(stdout)
        monkeypatch.setattr(gh.subprocess, "run", rec)
        return rec

    return _install


class TestWrapperArgv:
    def test_project_view_builds_argv_and_parses(self, wire) -> None:
        rec = wire(json.dumps({"id": "PVT_1", "title": "RM"}))
        out = gh_project_view("acme", 7)
        assert out == GithubProject(id="PVT_1", title="RM")
        assert rec.calls[0][1:] == ["project", "view", "7", "--owner", "acme", "--format", "json"]

    def test_item_list_returns_items_key(self, wire) -> None:
        rec = wire(json.dumps({"items": [{"id": "PVTI_1", "title": "g"}]}))
        out = gh_item_list("acme", 7, limit=3)
        assert out == [GithubItem(id="PVTI_1", title="g")]
        assert rec.calls[0][1:] == [
            "project",
            "item-list",
            "7",
            "--owner",
            "acme",
            "--format",
            "json",
            "--limit",
            "3",
        ]

    def test_item_list_defaults_empty_when_no_items(self, wire) -> None:
        wire(json.dumps({}))
        assert gh_item_list("acme", 7) == []

    def test_item_add_returns_item_id(self, wire) -> None:
        rec = wire(json.dumps({"id": "PVTI_new"}))
        assert gh_item_add("acme", 7, "https://x/1") == "PVTI_new"
        assert rec.calls[0][1:] == [
            "project",
            "item-add",
            "7",
            "--owner",
            "acme",
            "--url",
            "https://x/1",
            "--format",
            "json",
        ]

    def test_item_edit_text_branch(self, wire) -> None:
        rec = wire("{}")
        gh_item_edit("PVT_1", "PVTI_1", "F_1", text="hello")
        assert rec.calls[0][1:] == [
            "project",
            "item-edit",
            "--id",
            "PVTI_1",
            "--project-id",
            "PVT_1",
            "--field-id",
            "F_1",
            "--text",
            "hello",
        ]

    def test_item_edit_option_branch(self, wire) -> None:
        rec = wire("{}")
        gh_item_edit("PVT_1", "PVTI_1", "F_1", single_select_option_id="opt-9")
        assert rec.calls[0][-2:] == ["--single-select-option-id", "opt-9"]

    def test_item_edit_rejects_both(self, wire) -> None:
        wire("{}")
        with pytest.raises(ValueError, match="exactly one"):
            gh_item_edit("PVT_1", "PVTI_1", "F_1", text="x", single_select_option_id="y")

    def test_item_edit_rejects_neither(self, wire) -> None:
        wire("{}")
        with pytest.raises(ValueError, match="exactly one"):
            gh_item_edit("PVT_1", "PVTI_1", "F_1")

    def test_field_list_returns_fields_key(self, wire) -> None:
        rec = wire(json.dumps({"fields": [{"id": "F_1", "name": "Milknado Status"}]}))
        out = gh_field_list("acme", 7)
        assert out == [GithubField(id="F_1", name="Milknado Status")]
        assert rec.calls[0][1:] == [
            "project",
            "field-list",
            "7",
            "--owner",
            "acme",
            "--format",
            "json",
        ]

    def test_field_create_single_select_joins_options(self, wire) -> None:
        rec = wire(json.dumps({"id": "F_new"}))
        out = gh_field_create(7, "acme", "Milknado Status", ["Pending", "Done"])
        assert out == "F_new"
        assert rec.calls[0][1:] == [
            "project",
            "field-create",
            "7",
            "--owner",
            "acme",
            "--name",
            "Milknado Status",
            "--data-type",
            "SINGLE_SELECT",
            "--single-select-options",
            "Pending,Done",
        ]

    def test_field_create_text_uses_text_data_type(self, wire) -> None:
        rec = wire(json.dumps({"id": "F_txt"}))
        out = gh_field_create_text(7, "acme", "Milknado Harvest")
        assert out == "F_txt"
        assert rec.calls[0][1:] == [
            "project",
            "field-create",
            "7",
            "--owner",
            "acme",
            "--name",
            "Milknado Harvest",
            "--data-type",
            "TEXT",
        ]

    def test_issue_create_returns_stripped_url(self, wire) -> None:
        rec = wire("https://github.com/acme/repo/issues/3\n")
        url = gh_issue_create("acme", "repo", "Title", "Body")
        assert url == "https://github.com/acme/repo/issues/3"
        assert rec.calls[0][1:] == [
            "issue",
            "create",
            "--repo",
            "acme/repo",
            "--title",
            "Title",
            "--body",
            "Body",
        ]

    def test_issue_edit_body_argv(self, wire) -> None:
        rec = wire("")
        gh_issue_edit_body("https://x/3", "new body")
        assert rec.calls[0][1:] == ["issue", "edit", "https://x/3", "--body", "new body"]

    def test_issue_view_builds_argv_and_returns_typed_issue(self, wire) -> None:
        rec = wire(
            '{"title":"Bug","body":"Details","number":7,'
            '"url":"https://github.com/acme/app/issues/7"}'
        )

        issue = gh_issue_view("acme/app#7")

        assert issue == GithubIssue(
            title="Bug",
            body="Details",
            number=7,
            url="https://github.com/acme/app/issues/7",
        )
        assert rec.calls == [
            [
                GH_BIN,
                "issue",
                "view",
                "acme/app#7",
                "--json",
                "title,body,number,url",
            ]
        ]


class TestZeroGraphql:
    def test_no_wrapper_argv_contains_graphql_or_api(self, wire) -> None:
        # Acceptance 8: the entire transport is `gh project`/`gh issue` — never
        # a raw GraphQL/api call. Exercise every wrapper and scan the argv.
        rec = wire(json.dumps({"id": "x", "title": "t", "items": [], "fields": []}))
        gh_project_view("a", 1)
        gh_item_list("a", 1)
        gh_item_add("a", 1, "u")
        gh_item_edit("PVT", "PVTI", "F", text="t")
        gh_field_list("a", 1)
        gh_field_create(1, "a", "n", ["o"])
        gh_field_create_text(1, "a", "n")
        gh_issue_create("a", "r", "t", "b")
        gh_issue_edit_body("u", "b")
        tokens = [tok for call in rec.calls for tok in call]
        assert "graphql" not in tokens
        assert "api" not in tokens
