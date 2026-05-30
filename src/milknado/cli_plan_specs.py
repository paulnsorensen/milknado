"""Spec materialization helpers for the plan command."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import typer
from rich.console import Console

console = Console()

_ISSUE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _fetch_issue(issue_ref: str) -> dict[str, object]:
    """Fetch a single GitHub issue via `gh`; exits on error."""
    try:
        result = subprocess.run(
            ["gh", "issue", "view", issue_ref, "--json", "title,body,number,url"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        console.print("[red]`gh` CLI not found. Install GitHub CLI to use --issue.[/red]")
        raise typer.Exit(code=1) from None

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        console.print(f"[red]gh issue view {issue_ref} failed:[/red] {stderr}")
        raise typer.Exit(code=1)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        console.print(f"[red]gh returned invalid JSON for {issue_ref}: {exc}[/red]")
        raise typer.Exit(code=1) from None


def _slug_for(refs: list[str], issues: list[dict[str, object]]) -> str:
    numbers = [str(i.get("number")) for i in issues if i.get("number") is not None]
    source = "-".join(numbers) if numbers else "-".join(refs) or "issue"
    return _ISSUE_SLUG_RE.sub("-", source).strip("-") or "issue"


def _issue_title(issue: dict[str, object]) -> str:
    title = str(issue.get("title") or "").strip()
    if title:
        return title
    number = issue.get("number")
    return f"Issue {number}" if number is not None else "Issue"


def _render_single_issue(issue: dict[str, object]) -> str:
    title = _issue_title(issue)
    url = issue.get("url") or ""
    body = str(issue.get("body") or "").rstrip()
    header = f"# {title}\n"
    if url:
        header += f"\n> Source: {url}\n"
    return f"{header}\n{body}\n"


def _render_issue_section(issue: dict[str, object]) -> str:
    title = _issue_title(issue)
    number = issue.get("number")
    url = issue.get("url") or ""
    body = str(issue.get("body") or "").rstrip()
    heading = f"## #{number}: {title}" if number is not None else f"## {title}"
    lines = [heading]
    if url:
        lines.append(f"\n> Source: {url}")
    lines.append(f"\n{body}")
    return "\n".join(lines) + "\n"


def _render_multi_issue(issues: list[dict[str, object]]) -> str:
    refs = [f"#{i.get('number')}" for i in issues if i.get("number") is not None]
    combined_title = "Plan for issues " + ", ".join(refs) if refs else "Plan"
    sections = [f"# {combined_title}\n"]
    for issue in issues:
        sections.append(_render_issue_section(issue))
    return "\n".join(sections) + "\n"


def _render_issue_spec(issues: list[dict[str, object]]) -> str:
    if len(issues) == 1:
        return _render_single_issue(issues[0])
    return _render_multi_issue(issues)


def _materialize_issue_spec(issue_refs: list[str], project_root: Path) -> Path:
    """Fetch one or more GitHub issues and write them as a single spec .md file."""
    if not issue_refs:
        raise ValueError("issue_refs must not be empty")
    issues = [_fetch_issue(ref) for ref in issue_refs]

    issues_dir = project_root / ".milknado" / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    spec_path = issues_dir / f"issue-{_slug_for(issue_refs, issues)}.md"
    spec_path.write_text(_render_issue_spec(issues), encoding="utf-8")
    return spec_path


def _render_spec_section(spec_path: Path) -> str:
    body = spec_path.read_text(encoding="utf-8").rstrip()
    return f"## Spec: {spec_path.stem}\n\n> Source: {spec_path}\n\n{body}\n"


def _combined_title(spec_paths: list[Path], issues: list[dict[str, object]]) -> str:
    parts: list[str] = []
    if spec_paths:
        parts.append("specs " + ", ".join(sp.stem for sp in spec_paths))
    if issues:
        refs = [f"#{i.get('number')}" for i in issues if i.get("number") is not None]
        if refs:
            parts.append("issues " + ", ".join(refs))
    return "Plan for " + " + ".join(parts) if parts else "Plan"


def _combined_slug(spec_paths: list[Path], issues: list[dict[str, object]]) -> str:
    tokens: list[str] = [sp.stem for sp in spec_paths]
    tokens += [str(i.get("number")) for i in issues if i.get("number") is not None]
    source = "-".join(tokens) or "plan"
    return _ISSUE_SLUG_RE.sub("-", source).strip("-") or "plan"


def _materialize_combined_spec(
    spec_paths: list[Path],
    issue_refs: list[str],
    project_root: Path,
) -> Path:
    """Merge multiple specs and/or issues into one materialized spec .md."""
    issues = [_fetch_issue(ref) for ref in issue_refs]
    sections = [f"# {_combined_title(spec_paths, issues)}\n"]
    for sp in spec_paths:
        sections.append(_render_spec_section(sp))
    for issue in issues:
        sections.append(_render_issue_section(issue))

    issues_dir = project_root / ".milknado" / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    spec_path = issues_dir / f"plan-{_combined_slug(spec_paths, issues)}.md"
    spec_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return spec_path


def _resolve_plan_spec(
    spec_paths: list[Path],
    issue_refs: list[str],
    project_root: Path,
) -> Path:
    """Pick the right spec to feed the planner, materializing as needed."""
    if len(spec_paths) == 1 and not issue_refs:
        return spec_paths[0]
    if not spec_paths and issue_refs:
        return _materialize_issue_spec(issue_refs, project_root)
    return _materialize_combined_spec(spec_paths, issue_refs, project_root)
