"""Pure spec/issue ingestion helpers — no typer, no global state."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

_ISSUE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


class SpecIngestError(Exception):
    """Raised when spec/issue ingestion fails."""


def validate_spec_paths(raw_paths: list[str]) -> list[Path]:
    validated: list[Path] = []
    for p in raw_paths:
        path = Path(p).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise SpecIngestError(f"--spec file not found: {p}")
        if path.suffix.lower() != ".md":
            raise SpecIngestError(f"--spec must point to a .md file, got: {path}")
        validated.append(path)
    return validated


def fetch_issue(issue_ref: str) -> dict[str, object]:
    try:
        result = subprocess.run(
            ["gh", "issue", "view", issue_ref, "--json", "title,body,number,url"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise SpecIngestError(
            "`gh` CLI not found. Install GitHub CLI to use --issue."
        ) from None

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise SpecIngestError(f"gh issue view {issue_ref} failed: {stderr}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SpecIngestError(
            f"gh returned invalid JSON for {issue_ref}: {exc}"
        ) from None


def slug_for(refs: list[str], issues: list[dict[str, object]]) -> str:
    numbers = [str(i.get("number")) for i in issues if i.get("number") is not None]
    source = "-".join(numbers) if numbers else "-".join(refs) or "issue"
    return _ISSUE_SLUG_RE.sub("-", source).strip("-") or "issue"


def issue_title(issue: dict[str, object]) -> str:
    title = str(issue.get("title") or "").strip()
    if title:
        return title
    number = issue.get("number")
    return f"Issue {number}" if number is not None else "Issue"


def render_issue_section(issue: dict[str, object]) -> str:
    title = issue_title(issue)
    number = issue.get("number")
    url = issue.get("url") or ""
    body = str(issue.get("body") or "").rstrip()
    heading = f"## #{number}: {title}" if number is not None else f"## {title}"
    lines = [heading]
    if url:
        lines.append(f"\n> Source: {url}")
    lines.append(f"\n{body}")
    return "\n".join(lines) + "\n"


def render_single_issue(issue: dict[str, object]) -> str:
    title = issue_title(issue)
    url = issue.get("url") or ""
    body = str(issue.get("body") or "").rstrip()
    header = f"# {title}\n"
    if url:
        header += f"\n> Source: {url}\n"
    return f"{header}\n{body}\n"


def render_multi_issue(issues: list[dict[str, object]]) -> str:
    refs = [f"#{i.get('number')}" for i in issues if i.get("number") is not None]
    combined_title = "Plan for issues " + ", ".join(refs) if refs else "Plan"
    sections = [f"# {combined_title}\n"]
    for issue in issues:
        sections.append(render_issue_section(issue))
    return "\n".join(sections) + "\n"


def render_issue_spec(issues: list[dict[str, object]]) -> str:
    if len(issues) == 1:
        return render_single_issue(issues[0])
    return render_multi_issue(issues)


def derive_goal(spec_path: Path) -> str:
    """Extract the first H1 heading from a spec file as the planning goal."""
    try:
        text = spec_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SpecIngestError(
            f"--spec file is not valid UTF-8 text: {spec_path} ({exc})"
        ) from None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            if heading:
                return heading
    return spec_path.stem


def render_spec_section(spec_path: Path) -> str:
    body = spec_path.read_text(encoding="utf-8").rstrip()
    return f"## Spec: {spec_path.stem}\n\n> Source: {spec_path}\n\n{body}\n"


def combined_title(spec_paths: list[Path], issues: list[dict[str, object]]) -> str:
    parts: list[str] = []
    if spec_paths:
        parts.append("specs " + ", ".join(sp.stem for sp in spec_paths))
    if issues:
        refs = [f"#{i.get('number')}" for i in issues if i.get("number") is not None]
        if refs:
            parts.append("issues " + ", ".join(refs))
    return "Plan for " + " + ".join(parts) if parts else "Plan"


def combined_slug(spec_paths: list[Path], issues: list[dict[str, object]]) -> str:
    tokens: list[str] = [sp.stem for sp in spec_paths]
    tokens += [str(i.get("number")) for i in issues if i.get("number") is not None]
    source = "-".join(tokens) or "plan"
    return _ISSUE_SLUG_RE.sub("-", source).strip("-") or "plan"


def materialize_issue_spec(issue_refs: list[str], project_root: Path) -> Path:
    if not issue_refs:
        raise ValueError("issue_refs must not be empty")
    issues = [fetch_issue(ref) for ref in issue_refs]

    issues_dir = project_root / ".milknado" / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    spec_path = issues_dir / f"issue-{slug_for(issue_refs, issues)}.md"
    spec_path.write_text(render_issue_spec(issues), encoding="utf-8")
    return spec_path


def materialize_combined_spec(
    spec_paths: list[Path],
    issue_refs: list[str],
    project_root: Path,
) -> Path:
    issues = [fetch_issue(ref) for ref in issue_refs]
    sections = [f"# {combined_title(spec_paths, issues)}\n"]
    for sp in spec_paths:
        sections.append(render_spec_section(sp))
    for issue in issues:
        sections.append(render_issue_section(issue))

    issues_dir = project_root / ".milknado" / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    spec_path = issues_dir / f"plan-{combined_slug(spec_paths, issues)}.md"
    spec_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return spec_path
