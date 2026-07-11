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


class GhTransportError(RuntimeError):
    """A `gh` invocation failed — raised loud with the command and its stderr."""


def _gh_bin() -> str:
    path = shutil.which("gh")
    if path is None:
        raise GhTransportError(
            "the `gh` CLI is not installed or not on PATH; install GitHub CLI and run "
            "`gh auth refresh -s project`"
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
        raise GhTransportError(f"`gh {' '.join(args)}` failed: {exc.stderr.strip()}") from exc


def gh_preflight() -> None:
    """Verify gh is installed, authenticated, and holds the `project` token scope.

    Fails fast with a message instructing `gh auth refresh -s project` (Acceptance 6).
    """
    _gh_bin()
    result = subprocess.run(
        [_gh_bin(), "auth", "status"],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        raise GhTransportError(
            "gh is not authenticated; run `gh auth login` then "
            "`gh auth refresh -s project` to grant the Projects scope"
        )
    if "project" not in combined:
        raise GhTransportError(
            "gh token is missing the Projects scope; run `gh auth refresh -s project`"
        )


def gh_project_view(owner: str, number: int) -> dict:
    """Return the project's own metadata, including its `id` (PVT node id)."""
    result = _run(["project", "view", str(number), "--owner", owner, "--format", "json"])
    return json.loads(result.stdout)


def gh_item_list(owner: str, number: int, *, limit: int = 500) -> list[dict]:
    """Return the project's items: each `{id, title, body, type, url?}`.

    `id` is the ProjectV2 *item* node id (PVTI_...), the goal's github_ref.
    """
    result = _run(
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
        ]
    )
    return json.loads(result.stdout).get("items", [])


def gh_item_add(owner: str, number: int, issue_url: str) -> str:
    """Link an existing Issue as a project item; return the new item node id."""
    result = _run(
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
        ]
    )
    return json.loads(result.stdout)["id"]


def gh_item_edit(
    project_id: str,
    item_id: str,
    field_id: str,
    *,
    text: str | None = None,
    single_select_option_id: str | None = None,
) -> None:
    """Set one field value on a project item (one field per invocation)."""
    if (text is None) == (single_select_option_id is None):
        raise ValueError("gh_item_edit requires exactly one of text or single_select_option_id")
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
    if text is not None:
        args += ["--text", text]
    else:
        args += ["--single-select-option-id", single_select_option_id]  # type: ignore[list-item]
    _run(args)


def gh_field_list(owner: str, number: int) -> list[dict]:
    """Return the project's fields: each `{id, name, type, options?:[{id,name}]}`."""
    result = _run(["project", "field-list", str(number), "--owner", owner, "--format", "json"])
    return json.loads(result.stdout).get("fields", [])


def gh_field_create(number: int, owner: str, name: str, options: list[str]) -> dict:
    """Create a single-select field with ALL options in one call (create-once).

    Never adds options to an existing field — that path is GraphQL-only.
    """
    result = _run(
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
        ]
    )
    return json.loads(result.stdout)


def gh_field_create_text(number: int, owner: str, name: str) -> dict:
    """Create a free-text field (create-once, used for the harvest field)."""
    result = _run(
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
        ]
    )
    return json.loads(result.stdout)


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
    _run(["issue", "edit", issue_url, "--body", body])
