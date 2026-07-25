from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class IssueSource(Protocol):
    title: str
    body: str
    number: int | None
    url: str


IssueFetcher = Callable[[str], IssueSource]
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_plan_identifier(source: str, fallback: str) -> str:
    """Normalize a planning-owned materialized-spec identifier.

    The identifier preserves dots and underscores from spec stems. Blank input
    or input containing no permitted identifier characters returns the caller's
    fallback; planning has no length limit or collision disambiguation.
    """
    return _SLUG_RE.sub("-", source).strip("-") or fallback


def _issue_title(issue: IssueSource) -> str:
    return issue.title.strip() or (
        f"Issue {issue.number}" if issue.number is not None else "Issue"
    )


def _render_single_issue(issue: IssueSource) -> str:
    header = f"# {_issue_title(issue)}\n"
    if issue.url:
        header += f"\n> Source: {issue.url}\n"
    return f"{header}\n{issue.body.rstrip()}\n"


def _render_issue_section(issue: IssueSource) -> str:
    title = _issue_title(issue)
    heading = f"## #{issue.number}: {title}" if issue.number is not None else f"## {title}"
    lines = [heading]
    if issue.url:
        lines.append(f"\n> Source: {issue.url}")
    lines.append(f"\n{issue.body.rstrip()}")
    return "\n".join(lines) + "\n"


def _write_materialized(project_root: Path, name: str, content: str) -> Path:
    directory = project_root / ".milknado" / "issues"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def materialize_issue_spec(
    issue_refs: list[str],
    project_root: Path,
    fetch_issue: IssueFetcher,
) -> Path:
    if not issue_refs:
        raise ValueError("issue_refs must not be empty")
    issues = [fetch_issue(ref) for ref in issue_refs]
    numbers = [str(issue.number) for issue in issues if issue.number is not None]
    slug = normalize_plan_identifier("-".join(numbers or issue_refs), "issue")
    if len(issues) == 1:
        content = _render_single_issue(issues[0])
    else:
        refs = [f"#{issue.number}" for issue in issues if issue.number is not None]
        title = "Plan for issues " + ", ".join(refs) if refs else "Plan"
        content = "\n".join([f"# {title}\n", *map(_render_issue_section, issues)]) + "\n"
    return _write_materialized(project_root, f"issue-{slug}.md", content)


def _render_spec_section(spec_path: Path) -> str:
    body = spec_path.read_text(encoding="utf-8").rstrip()
    return f"## Spec: {spec_path.stem}\n\n> Source: {spec_path}\n\n{body}\n"


def materialize_combined_spec(
    spec_paths: list[Path],
    issue_refs: list[str],
    project_root: Path,
    fetch_issue: IssueFetcher,
) -> Path:
    issues = [fetch_issue(ref) for ref in issue_refs]
    parts = ["specs " + ", ".join(path.stem for path in spec_paths)] if spec_paths else []
    numbers = [f"#{issue.number}" for issue in issues if issue.number is not None]
    if numbers:
        parts.append("issues " + ", ".join(numbers))
    sections = [f"# {'Plan for ' + ' + '.join(parts) if parts else 'Plan'}\n"]
    sections.extend(_render_spec_section(path) for path in spec_paths)
    sections.extend(_render_issue_section(issue) for issue in issues)
    tokens = [path.stem for path in spec_paths]
    tokens.extend(str(issue.number) for issue in issues if issue.number is not None)
    slug = normalize_plan_identifier("-".join(tokens), "plan")
    return _write_materialized(project_root, f"plan-{slug}.md", "\n".join(sections) + "\n")


def resolve_plan_spec(
    spec_paths: list[Path],
    issue_refs: list[str],
    project_root: Path,
    fetch_issue: IssueFetcher,
) -> Path:
    if len(spec_paths) == 1 and not issue_refs:
        return spec_paths[0]
    if not spec_paths and issue_refs:
        return materialize_issue_spec(issue_refs, project_root, fetch_issue)
    return materialize_combined_spec(spec_paths, issue_refs, project_root, fetch_issue)


def derive_goal(spec_path: Path) -> str:
    text = spec_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and (heading := stripped[2:].strip()):
            return heading
    return spec_path.stem
