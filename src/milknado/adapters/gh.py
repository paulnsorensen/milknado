"""GitHub Projects v2 transport — thin `gh project` subprocess wrappers.

The only place milknado shells out to `gh`. Zero GraphQL: every operation goes
through a `gh project` / `gh issue` subcommand (verified in the research artifact
`.cheese/research/gh-projectv2-cli-transport/`). All wrappers fail fast and loud.

The `gh` binary is resolved via `shutil.which` and invoked by absolute path so
this repo's rtk shell hook — which rewrites a bare `gh project ...` into an
unsupported `rtk gh project` form — never intercepts the call.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from typing import Annotated, TypeVar, cast, final

import msgspec

from milknado.domains.github import (
    GithubField,
    GithubIssue,
    GithubItem,
    GithubProject,
)


class GhTransportError(RuntimeError):
    """A `gh` invocation failed — raised loud with the command and its stderr."""


class _IdResponse(msgspec.Struct, frozen=True, kw_only=True):
    id: Annotated[str, msgspec.Meta(min_length=1)]


class _ItemListResponse(msgspec.Struct, frozen=True, kw_only=True):
    items: list[GithubItem] = msgspec.field(default_factory=list)


class _FieldListResponse(msgspec.Struct, frozen=True, kw_only=True):
    fields: list[GithubField] = msgspec.field(default_factory=list)


ResponseModel = TypeVar("ResponseModel")


def _gh_bin() -> str:
    path = shutil.which("gh")
    if path is None:
        raise GhTransportError(
            "the `gh` CLI is not installed or not on PATH; install GitHub CLI and run "
            + "`gh auth refresh -s project`"
        )
    return path


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke `gh <args>` by absolute path; raise GhTransportError on failure."""
    try:
        return subprocess.run(
            [_gh_bin(), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = cast(str | None, exc.stderr)
        raise GhTransportError(f"`gh {' '.join(args)}` failed: {(stderr or '').strip()}") from exc


def _run_json(args: list[str]) -> object:
    """Run `gh <args>` and parse its JSON response."""
    result = _run(args)
    try:
        return cast(object, json.loads(result.stdout))
    except json.JSONDecodeError as exc:
        raise GhTransportError(
            f"`gh {' '.join(args)}` returned unparseable JSON: {result.stdout.strip()[:200]!r}"
        ) from exc


def _run_model(args: list[str], model: type[ResponseModel]) -> ResponseModel:
    data = _run_json(args)
    try:
        return msgspec.convert(data, type=model, strict=True)
    except msgspec.ValidationError as exc:
        command = " ".join(args[:2])
        raise GhTransportError(f"`gh {command}` response is malformed: {exc}") from exc


def gh_preflight() -> None:
    """Verify gh is installed, authenticated, and holds the `project` token scope.

    Fails fast with a message instructing `gh auth refresh -s project` (Acceptance 6).
    """
    gh_bin = _gh_bin()
    result = subprocess.run(
        [gh_bin, "auth", "status"],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        raise GhTransportError(
            "gh is not authenticated; run `gh auth login` then "
            + "`gh auth refresh -s project` to grant the Projects scope"
        )
    if "'project'" not in combined:
        raise GhTransportError(
            "gh token is missing the Projects scope; run `gh auth refresh -s project`"
        )


def gh_project_view(owner: str, number: int) -> GithubProject:
    """Return the project's validated metadata."""
    return _run_model(
        ["project", "view", str(number), "--owner", owner, "--format", "json"],
        GithubProject,
    )


def gh_item_list(owner: str, number: int, *, limit: int = 500) -> list[GithubItem]:
    """Return validated Project items."""
    response = _run_model(
        [
            "project",
            "item-list",
            str(number),
            "--owner",
            owner,
            "--format",
            "json",
            "--limit",
            str(limit),
        ],
        _ItemListResponse,
    )
    return response.items


def gh_item_add(owner: str, number: int, issue_url: str) -> str:
    """Link an existing Issue as a Project item."""
    response = _run_model(
        [
            "project",
            "item-add",
            str(number),
            "--owner",
            owner,
            "--url",
            issue_url,
            "--format",
            "json",
        ],
        _IdResponse,
    )
    return response.id


def gh_item_edit(
    project_id: str,
    item_id: str,
    field_id: str,
    *,
    text: str | None = None,
    single_select_option_id: str | None = None,
) -> None:
    """Set one field value on a Project item."""
    args = [
        "project",
        "item-edit",
        "--id",
        item_id,
        "--project-id",
        project_id,
        "--field-id",
        field_id,
    ]
    match text, single_select_option_id:
        case str(), None:
            args.extend(("--text", text))
        case None, str():
            args.extend(("--single-select-option-id", single_select_option_id))
        case _:
            raise ValueError(
                "gh_item_edit requires exactly one of text or single_select_option_id"
            )
    _ = _run(args)


def gh_field_list(owner: str, number: int) -> list[GithubField]:
    """Return validated Project fields."""
    response = _run_model(
        ["project", "field-list", str(number), "--owner", owner, "--format", "json"],
        _FieldListResponse,
    )
    return response.fields


def gh_field_create(number: int, owner: str, name: str, options: Sequence[str]) -> str:
    """Create a single-select field with all options in one call."""
    response = _run_model(
        [
            "project",
            "field-create",
            str(number),
            "--owner",
            owner,
            "--name",
            name,
            "--data-type",
            "SINGLE_SELECT",
            "--single-select-options",
            ",".join(options),
        ],
        _IdResponse,
    )
    return response.id


def gh_field_create_text(number: int, owner: str, name: str) -> str:
    """Create a free-text field."""
    response = _run_model(
        [
            "project",
            "field-create",
            str(number),
            "--owner",
            owner,
            "--name",
            name,
            "--data-type",
            "TEXT",
        ],
        _IdResponse,
    )
    return response.id


def gh_issue_view(ref: str) -> GithubIssue:
    """Return validated identity and content for one GitHub issue."""
    return _run_model(
        ["issue", "view", ref, "--json", "title,body,number,url"],
        GithubIssue,
    )


def gh_issue_create(owner: str, repo: str, title: str, body: str) -> str:
    """Create an Issue and return its URL (used for wiki-origin projection)."""
    result = _run(
        [
            "issue",
            "create",
            "--repo",
            f"{owner}/{repo}",
            "--title",
            title,
            "--body",
            body,
        ]
    )
    return result.stdout.strip()


def gh_issue_edit_body(issue_url: str, body: str) -> None:
    """Overwrite an Issue's body (wiki-origin body mirror, overwritten each export)."""
    _ = _run(["issue", "edit", issue_url, "--body", body])


@final
class GhProjectAdapter:
    preflight = staticmethod(gh_preflight)
    project_view = staticmethod(gh_project_view)
    item_list = staticmethod(gh_item_list)
    item_add = staticmethod(gh_item_add)
    item_edit = staticmethod(gh_item_edit)
    field_list = staticmethod(gh_field_list)
    field_create = staticmethod(gh_field_create)
    field_create_text = staticmethod(gh_field_create_text)
    issue_create = staticmethod(gh_issue_create)
    issue_edit_body = staticmethod(gh_issue_edit_body)
