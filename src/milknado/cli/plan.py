"""Plan command — thin I/O parsing over milknado.app.plan.

Parses --spec/--issue, resolves the goal, then hands off to the application-layer
planning policy in ``milknado.app.plan``. The critic, interactive/non-interactive
runners, planner subprocess port, and planner adapter wiring all live there; the
symbols the test-suite exercises directly are re-exported below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from milknado.app.plan import (
    CriticVerdict,
    _exec_plan_interactive,
    _exec_plan_non_interactive,
    _parse_critic_output,
    _plan_exit_code,
    _plan_summary,
    _PlanningSubprocess,
    _run_plan_with_critic,
    build_planner,
    execute_plan,
    run_plan_critic,
)
from milknado.cli._helpers import _ensure_db, _load_or_default

console = Console()

__all__ = [
    "CriticVerdict",
    "_PlanningSubprocess",
    "_derive_goal",
    "_exec_plan_interactive",
    "_exec_plan_non_interactive",
    "_fetch_issue",
    "_parse_critic_output",
    "_plan_exit_code",
    "_plan_summary",
    "_run_plan_with_critic",
    "plan",
    "run_plan_critic",
]


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


def _fetch_issue(issue_ref: str):
    from milknado.adapters.gh import GhTransportError, gh_issue_view

    try:
        return gh_issue_view(issue_ref)
    except GhTransportError as exc:
        console.print(f"[red]gh issue view {issue_ref} failed:[/red] {exc}")
        raise typer.Exit(code=1) from None


def _resolve_plan_spec(
    spec_paths: list[Path],
    issue_refs: list[str],
    project_root: Path,
) -> Path:
    from milknado.domains.planning.source_material import resolve_plan_spec

    return resolve_plan_spec(spec_paths, issue_refs, project_root, _fetch_issue)


def _derive_goal(spec_path: Path) -> str:
    from milknado.domains.planning.source_material import derive_goal

    try:
        return derive_goal(spec_path)
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(
            f"--spec file is not valid UTF-8 text: {spec_path} ({exc})"
        ) from None


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
        planner = build_planner(graph, project_root, config)
        console.print(f"[bold]Planning:[/bold] {goal}")
        execute_plan(
            planner, goal, project_root, effective_spec, interactive, max_iterations, config
        )
    finally:
        graph.close()
