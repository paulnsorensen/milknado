"""plan command and all supporting helpers (issue resolution, spec materialization)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from milknado.domains.planning import Planner
    from milknado.domains.planning.planner import PlanResult

import typer
from rich.console import Console

from milknado._cli_helpers import _ensure_db, _load_or_default

console = Console()

_ISSUE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ── CSV helpers ───────────────────────────────────────────────────────────────


def _flatten_csv(items: list[str] | None) -> list[str]:
    if not items:
        return []
    out: list[str] = []
    for raw in items:
        for piece in raw.split(","):
            stripped = piece.strip()
            if stripped:
                out.append(stripped)
    return out


def _validate_spec_paths(raw_paths: list[str]) -> list[Path]:
    validated: list[Path] = []
    for p in raw_paths:
        path = Path(p).expanduser().resolve()
        if not path.exists() or not path.is_file():
            console.print(f"[red]--spec file not found: {p}[/red]")
            raise typer.Exit(code=1)
        if path.suffix.lower() != ".md":
            console.print(f"[red]--spec must point to a .md file, got: {path}[/red]")
            raise typer.Exit(code=1)
        validated.append(path)
    return validated


# ── GitHub issue fetching ─────────────────────────────────────────────────────


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


# ── Issue rendering ───────────────────────────────────────────────────────────


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


# ── Spec materialization ──────────────────────────────────────────────────────


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


def _derive_goal(spec_path: Path) -> str:
    try:
        text = spec_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(
            f"--spec file is not valid UTF-8 text: {spec_path} ({exc})"
        ) from None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            if heading:
                return heading
    return spec_path.stem


# ── Plan result helpers ───────────────────────────────────────────────────────


def _plan_summary(result: PlanResult) -> str:
    return (
        f"Planned {result.change_count} changes → {result.batch_count} batches"
        f" ({result.oversized_count} oversized), solver={result.solver_status};"
        f" {result.nodes_created} Mikado nodes created"
    )


def _plan_exit_code(result: PlanResult) -> int:
    if result.solver_status == "INFEASIBLE":
        return 1
    if result.solver_status == "NO_MANIFEST":
        return 1
    if result.solver_status in ("OPTIMAL", "FEASIBLE"):
        return 0
    if result.solver_status == "UNKNOWN" and result.batch_count >= 1:
        return 0
    if not result.success:
        return result.exit_code
    return 0


def _plan_iteration_summary(iteration: int, result: PlanResult) -> str:
    return f"[bold]Plan iteration {iteration}[/bold]: {_plan_summary(result)}"


def _build_replan_goal(base_goal: str, feedback: str, prior_result: PlanResult) -> str:
    return (
        f"{base_goal}\n\n"
        "## User revision request\n"
        f"{feedback.strip()}\n\n"
        "## Prior planner result\n"
        f"- solver_status: {prior_result.solver_status or 'UNKNOWN'}\n"
        f"- changes: {prior_result.change_count}\n"
        f"- batches: {prior_result.batch_count}\n"
        "Revise the plan manifest to address this feedback while preserving valid parts."
    )


def _prompt_plan_action() -> str:
    console.print("\n[bold]Choose next step:[/bold]")
    console.print("  1) Accept plan")
    console.print("  2) Revise with feedback")
    console.print("  3) Cancel")
    while True:
        choice = typer.prompt("Selection", show_choices=False).strip()
        if choice in {"1", "2", "3"}:
            return choice
        console.print("[yellow]Please enter 1, 2, or 3.[/yellow]")


# ── Extracted non-interactive / interactive runners ───────────────────────────


def _exec_plan_non_interactive(
    planner: Planner,
    goal: str,
    project_root: Path,
    effective_spec: Path,
) -> None:
    result: PlanResult = planner.launch(goal, project_root, spec_path=effective_spec)
    console.print(_plan_summary(result))
    exit_code = _plan_exit_code(result)
    if result.solver_status == "UNKNOWN" and result.batch_count >= 1:
        Console(stderr=True).print(
            "[yellow]Warning: solver returned UNKNOWN — results may be suboptimal[/yellow]"
        )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def _exec_plan_interactive(
    planner: Planner,
    goal: str,
    project_root: Path,
    effective_spec: Path,
    max_iterations: int,
) -> None:
    current_goal = goal
    for iteration in range(1, max_iterations + 1):
        result: PlanResult = planner.launch(current_goal, project_root, spec_path=effective_spec)
        console.print(_plan_iteration_summary(iteration, result))
        if result.context_path is not None:
            console.print(f"[dim]Planner context: {result.context_path}[/dim]")
        exit_code = _plan_exit_code(result)
        if result.solver_status == "UNKNOWN" and result.batch_count >= 1:
            Console(stderr=True).print(
                "[yellow]Warning: solver returned UNKNOWN — results may be suboptimal[/yellow]"
            )
        if exit_code != 0:
            console.print("[red]Planner output was invalid for execution.[/red]")
        action = _prompt_plan_action()
        if action == "1":
            if exit_code != 0:
                console.print(
                    "[yellow]Accepting current plan despite invalid planner status.[/yellow]"
                )
            return
        if action == "3":
            raise typer.Exit(code=1)
        feedback = typer.prompt("What should change in the plan?").strip()
        if not feedback:
            console.print(
                "[yellow]Empty feedback; keeping original goal for next iteration.[/yellow]"
            )
            continue
        current_goal = _build_replan_goal(goal, feedback, result)
    console.print(f"[red]Reached max iterations ({max_iterations}) without acceptance.[/red]")
    raise typer.Exit(code=1)


# ── plan command ──────────────────────────────────────────────────────────────


def plan(
    spec: Annotated[
        list[str] | None,
        typer.Option(
            "--spec",
            help=(
                "Spec .md file(s). Repeat the flag or pass comma-separated paths "
                "to combine multiple specs."
            ),
        ),
    ] = None,
    issue: Annotated[
        list[str] | None,
        typer.Option(
            "--issue",
            help=(
                "GitHub issue ref (number, owner/repo#123, or URL) fetched via gh. "
                "Repeat the flag or pass comma-separated refs to combine issues."
            ),
        ),
    ] = None,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive/--no-interactive",
            help="Iterate on planning with accept/revise/cancel prompts.",
        ),
    ] = False,
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            min=1,
            help="Maximum planner iterations in interactive mode.",
        ),
    ] = 5,
) -> None:
    """Launch interactive planning session to decompose one or more specs/issues."""
    from milknado.adapters.crg import CrgAdapter
    from milknado.domains.planning import Planner

    spec_paths = _validate_spec_paths(_flatten_csv(spec))
    issue_refs = _flatten_csv(issue)
    if not spec_paths and not issue_refs:
        console.print("[red]Provide --spec or --issue.[/red]")
        raise typer.Exit(code=1)

    project_root = project_root.resolve()
    effective_spec = _resolve_plan_spec(spec_paths, issue_refs, project_root)

    goal = _derive_goal(effective_spec)
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)

    try:
        crg = CrgAdapter(project_root)
        planner = Planner(
            graph,
            crg,
            config.planning_agent,
            planning_validation_hook=config.planning_validation_hook,
        )
        console.print(f"[bold]Planning:[/bold] {goal}")
        if not interactive:
            _exec_plan_non_interactive(planner, goal, project_root, effective_spec)
        else:
            _exec_plan_interactive(planner, goal, project_root, effective_spec, max_iterations)
    finally:
        graph.close()
