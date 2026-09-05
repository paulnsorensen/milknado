"""`adapters/gh.py` fails loudly on transport errors and validates every shaped
GitHub response with its Pydantic boundary model (Acceptance 6).

subprocess.run is fully mocked so these tests never touch a real `gh`.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from typing import NoReturn

import pytest

from milknado.adapters.gh import (
    GhTransportError,
    gh_field_list,
    gh_issue_view,
    gh_item_add,
    gh_preflight,
    gh_project_view,
)

GH_BIN = "/usr/bin/gh"


def _gh_binary(_name: str) -> str:
    return GH_BIN


def _missing_gh_binary(_name: str) -> None:
    return None


def _run_unauthenticated(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 1, stdout="", stderr="no auth")


def _run_missing_project_scope(
    *_args: object, **_kwargs: object
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout="Logged in with scopes: repo", stderr="")


def _run_authenticated(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout="Token scopes: 'repo', 'project'", stderr="")


def _run_read_only_scope(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout="Token scopes: 'read:project'", stderr="")


class Recorder:
    """Stand-in for subprocess.run that records argv and returns canned stdout."""

    def __init__(self, stdout: str = "") -> None:
        self.calls: list[list[str]] = []
        self.stdout: str = stdout

    def __call__(self, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=self.stdout, stderr="")


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Recorder]:
    """Pin `gh` on PATH and swap subprocess.run for a Recorder factory."""

    def _which(_name: str) -> str:
        return GH_BIN

    monkeypatch.setattr("milknado.adapters.gh.shutil.which", _which)

    def _install(stdout: str = "") -> Recorder:
        rec = Recorder(stdout)
        monkeypatch.setattr("milknado.adapters.gh.subprocess.run", rec)
        return rec

    return _install


class TestRunJson:
    def test_unparseable_json_raises_transport_error(
        self, wire: Callable[[str], Recorder]
    ) -> None:
        _ = wire("not json{")
        with pytest.raises(GhTransportError, match="unparseable JSON"):
            _ = gh_project_view("acme", 7)

    def test_non_object_json_raises_transport_error(self, wire: Callable[[str], Recorder]) -> None:
        _ = wire("[]")
        with pytest.raises(GhTransportError, match="response is malformed"):
            _ = gh_project_view("acme", 7)

    def test_item_add_missing_id_raises_transport_error(
        self, wire: Callable[[str], Recorder]
    ) -> None:
        _ = wire(json.dumps({"foo": 1}))
        with pytest.raises(GhTransportError, match=r"(?s)response is malformed.*id"):
            _ = gh_item_add("acme", 7, "https://x/1")


class TestIssueViewValidation:
    def test_missing_required_field_raises_transport_error(
        self, wire: Callable[[str], Recorder]
    ) -> None:
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
            _ = gh_issue_view("acme/app#7")

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
    def test_missing_id_raises_transport_error(self, wire: Callable[[str], Recorder]) -> None:
        _ = wire(json.dumps({"title": "RM"}))
        with pytest.raises(GhTransportError, match=r"(?s)response is malformed.*id"):
            _ = gh_project_view("acme", 7)

    def test_empty_id_raises_transport_error(self, wire: Callable[[str], Recorder]) -> None:
        _ = wire(json.dumps({"id": "", "title": "RM"}))
        with pytest.raises(GhTransportError, match=r"(?s)response is malformed.*id"):
            _ = gh_project_view("acme", 7)

    def test_missing_title_raises_transport_error(self, wire: Callable[[str], Recorder]) -> None:
        _ = wire(json.dumps({"id": "PVT_1"}))
        with pytest.raises(GhTransportError, match=r"(?s)response is malformed.*title"):
            _ = gh_project_view("acme", 7)


class TestFieldListValidation:
    def test_field_missing_id_raises_transport_error(
        self, wire: Callable[[str], Recorder]
    ) -> None:
        _ = wire(json.dumps({"fields": [{"name": "Milknado Status"}]}))
        with pytest.raises(GhTransportError, match=r"(?s)response is malformed.*id"):
            _ = gh_field_list("acme", 7)

    def test_field_missing_name_raises_transport_error(
        self, wire: Callable[[str], Recorder]
    ) -> None:
        _ = wire(json.dumps({"fields": [{"id": "F_1"}]}))
        with pytest.raises(GhTransportError, match=r"(?s)response is malformed.*name"):
            _ = gh_field_list("acme", 7)


class TestPreflight:
    def test_missing_gh_binary_raises_with_refresh_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("milknado.adapters.gh.shutil.which", _missing_gh_binary)
        with pytest.raises(GhTransportError, match="gh auth refresh -s project"):
            gh_preflight()

    def test_unauthenticated_raises_refresh_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("milknado.adapters.gh.shutil.which", _gh_binary)
        monkeypatch.setattr(
            "milknado.adapters.gh.subprocess.run",
            _run_unauthenticated,
        )
        with pytest.raises(GhTransportError, match="gh auth refresh -s project"):
            gh_preflight()

    def test_missing_project_scope_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("milknado.adapters.gh.shutil.which", _gh_binary)
        monkeypatch.setattr(
            "milknado.adapters.gh.subprocess.run",
            _run_missing_project_scope,
        )
        with pytest.raises(GhTransportError, match="Projects scope"):
            gh_preflight()

    def test_authenticated_with_scope_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("milknado.adapters.gh.shutil.which", _gh_binary)
        monkeypatch.setattr(
            "milknado.adapters.gh.subprocess.run",
            _run_authenticated,
        )
        gh_preflight()  # no raise

    def test_read_only_project_scope_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("milknado.adapters.gh.shutil.which", _gh_binary)
        monkeypatch.setattr(
            "milknado.adapters.gh.subprocess.run",
            _run_read_only_scope,
        )
        with pytest.raises(GhTransportError, match="Projects scope"):
            gh_preflight()


class TestRunFailure:
    def test_called_process_error_becomes_transport_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("milknado.adapters.gh.shutil.which", _gh_binary)

        def _boom(*_a: object, **_k: object) -> NoReturn:
            raise subprocess.CalledProcessError(1, "gh", stderr="boom detail")

        monkeypatch.setattr("milknado.adapters.gh.subprocess.run", _boom)
        with pytest.raises(GhTransportError, match="boom detail"):
            _ = gh_project_view("a", 1)
