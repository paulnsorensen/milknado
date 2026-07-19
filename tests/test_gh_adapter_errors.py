"""`adapters/gh.py` transport: preflight fails loud on missing gh/auth/scope,
`_run_json` rejects unparseable or non-object JSON, and each wrapper that
trusts a shaped reply (`gh project view`/`field-list`) validates the keys it
depends on rather than passing through a malformed payload (Acceptance 6).

subprocess.run is fully mocked so these tests never touch a real `gh`.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from milknado.adapters import gh
from milknado.adapters.gh import (
    GhTransportError,
    gh_field_list,
    gh_issue_view,
    gh_item_add,
    gh_preflight,
    gh_project_view,
)

GH_BIN = "/usr/bin/gh"


class Recorder:
    """Stand-in for subprocess.run that records argv and returns canned stdout."""

    def __init__(self, stdout: str = "") -> None:
        self.calls: list[list[str]] = []
        self.stdout = stdout

    def __call__(self, args, **_kwargs):  # noqa: ANN001, ANN002
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


class TestRunJson:
    def test_unparseable_json_raises_transport_error(self, wire) -> None:  # noqa: ANN001
        wire("not json{")
        with pytest.raises(GhTransportError, match="unparseable JSON"):
            gh_project_view("acme", 7)

    def test_non_object_json_raises_transport_error(self, wire) -> None:  # noqa: ANN001
        wire("[]")
        with pytest.raises(GhTransportError, match="non-object JSON"):
            gh_project_view("acme", 7)

    def test_item_add_missing_id_raises_transport_error(self, wire) -> None:  # noqa: ANN001
        wire(json.dumps({"foo": 1}))
        with pytest.raises(GhTransportError, match="has no `id`"):
            gh_item_add("acme", 7, "https://x/1")


class TestIssueViewValidation:
    def test_missing_required_field_raises_transport_error(self, wire) -> None:  # noqa: ANN001
        recorder = wire(
            json.dumps(
                {
                    "body": "Issue body",
                    "number": 7,
                    "url": "https://github.example/acme/app/issues/7",
                }
            )
        )

        with pytest.raises(GhTransportError, match=r"`gh issue view` response is malformed"):
            gh_issue_view("acme/app#7")

        assert recorder.calls == [
            [
                GH_BIN,
                "issue",
                "view",
                "acme/app#7",
                "--json",
                "title,body,number,url",
            ]
        ]


class TestProjectViewValidation:
    def test_missing_id_raises_transport_error(self, wire) -> None:  # noqa: ANN001
        wire(json.dumps({"title": "RM"}))
        with pytest.raises(GhTransportError, match="no valid `id`"):
            gh_project_view("acme", 7)

    def test_empty_id_raises_transport_error(self, wire) -> None:  # noqa: ANN001
        wire(json.dumps({"id": "", "title": "RM"}))
        with pytest.raises(GhTransportError, match="no valid `id`"):
            gh_project_view("acme", 7)

    def test_missing_title_raises_transport_error(self, wire) -> None:  # noqa: ANN001
        wire(json.dumps({"id": "PVT_1"}))
        with pytest.raises(GhTransportError, match="no `title`"):
            gh_project_view("acme", 7)


class TestFieldListValidation:
    def test_field_missing_id_raises_transport_error(self, wire) -> None:  # noqa: ANN001
        wire(json.dumps({"fields": [{"name": "Milknado Status"}]}))
        with pytest.raises(GhTransportError, match="no valid `id`"):
            gh_field_list("acme", 7)

    def test_field_missing_name_raises_transport_error(self, wire) -> None:  # noqa: ANN001
        wire(json.dumps({"fields": [{"id": "F_1"}]}))
        with pytest.raises(GhTransportError, match="no valid `name`"):
            gh_field_list("acme", 7)


class TestPreflight:
    def test_missing_gh_binary_raises_with_refresh_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gh.shutil, "which", lambda _name: None)
        with pytest.raises(GhTransportError, match="gh auth refresh -s project"):
            gh_preflight()

    def test_unauthenticated_raises_refresh_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gh.shutil, "which", lambda _name: GH_BIN)
        monkeypatch.setattr(
            gh.subprocess,
            "run",
            lambda *_a, **_k: subprocess.CompletedProcess([], 1, stdout="", stderr="no auth"),
        )
        with pytest.raises(GhTransportError, match="gh auth refresh -s project"):
            gh_preflight()

    def test_missing_project_scope_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gh.shutil, "which", lambda _name: GH_BIN)
        monkeypatch.setattr(
            gh.subprocess,
            "run",
            lambda *_a, **_k: subprocess.CompletedProcess(
                [], 0, stdout="Logged in with scopes: repo", stderr=""
            ),
        )
        with pytest.raises(GhTransportError, match="Projects scope"):
            gh_preflight()

    def test_authenticated_with_scope_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gh.shutil, "which", lambda _name: GH_BIN)
        monkeypatch.setattr(
            gh.subprocess,
            "run",
            lambda *_a, **_k: subprocess.CompletedProcess(
                [], 0, stdout="Token scopes: 'repo', 'project'", stderr=""
            ),
        )
        gh_preflight()  # no raise

    def test_read_only_project_scope_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gh.shutil, "which", lambda _name: GH_BIN)
        monkeypatch.setattr(
            gh.subprocess,
            "run",
            lambda *_a, **_k: subprocess.CompletedProcess(
                [], 0, stdout="Token scopes: 'read:project'", stderr=""
            ),
        )
        with pytest.raises(GhTransportError, match="Projects scope"):
            gh_preflight()


class TestRunFailure:
    def test_called_process_error_becomes_transport_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gh.shutil, "which", lambda _name: GH_BIN)

        def _boom(*_a, **_k):
            raise subprocess.CalledProcessError(1, "gh", stderr="boom detail")

        monkeypatch.setattr(gh.subprocess, "run", _boom)
        with pytest.raises(GhTransportError, match="boom detail"):
            gh_project_view("a", 1)
