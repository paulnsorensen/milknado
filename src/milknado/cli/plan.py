"""Plan CLI veneer; policy and adapter wiring live in milknado.app.plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from rich.console import Console

import milknado.app.plan as plan_policy
from milknado.cli._helpers import (
    DEFAULT_PROJECT_ROOT,
    typer_option,
)
from milknado.cli._helpers import (
    ensure_db as _ensure_db,
)
from milknado.cli._helpers import (
    load_or_default as _load_or_default,
)

if TYPE_CHECKING:
    from milknado.domains.common import MilknadoConfig
    from milknado.domains.planning import Planner, PlanResult

console = Console()


@dataclass
class _FetchedIssue:
    title: str
    body: str
    number: int | None
    url: str


CriticVerdict = plan_policy.CriticVerdict
PlanCriticError = plan_policy.PlanCriticError
_PlanningSubprocess = plan_policy._PlanningSubprocess
_parse_critic_output = plan_policy._parse_critic_output
_plan_exit_code = plan_policy._plan_exit_code
_plan_summary = plan_policy._plan_summary
_run_plan_with_critic = plan_policy._run_plan_with_critic
build_planner = plan_policy.build_planner
run_plan_critic = plan_policy.run_plan_critic

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
    "execute_plan",
    "plan",
    "run_plan_critic",
]


def _plan_iteration_summary(iteration: int, result: PlanResult) -> str:
    return f"[bold]Plan iteration {iteration}[/bold]: {_plan_summary(result)}"


def _prompt_plan_action() -> str:
    console.print("\n[bold]Choose next step:[/bold]")
    console.print("  1) Accept plan")
    console.print("  2) Revise with feedback")
    console.print("  3) Cancel")
    while True:
        choice = cast(str, typer.prompt("Selection", show_choices=False)).strip()
        if choice in {"1", "2", "3"}:
            return choice
        console.print("[yellow]Please enter 1, 2, or 3.[/yellow]")


def _exec_plan_non_interactive(
    planner: Planner,
    goal: str,
    project_root: Path,
    effective_spec: Path,
    cfg: MilknadoConfig,
) -> None:
    try:
        result, verdict = _run_plan_with_critic(planner, goal, project_root, effective_spec, cfg)
    except PlanCriticError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(_plan_summary(result))
    if verdict is not None and not verdict.approved:
        console.print(
            f"[red]Plan critic did not approve within {cfg.plan_review_max_rounds} "
            + f"round(s):[/red] {verdict.feedback}"
        )
        raise typer.Exit(code=1)
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
    cfg: MilknadoConfig,
) -> None:
    current_goal = goal
    for iteration in range(1, max_iterations + 1):
        try:
            result, verdict = _run_plan_with_critic(
                planner, current_goal, project_root, effective_spec, cfg
            )
        except PlanCriticError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None
        console.print(_plan_iteration_summary(iteration, result))
        if result.context_path is not None:
            console.print(f"[dim]Planner context: {result.context_path}[/dim]")
        if verdict is not None and not verdict.approved:
            console.print(
                f"[yellow]Plan critic did not approve within {cfg.plan_review_max_rounds} "
                + f"round(s); falling back to manual review:[/yellow] {verdict.feedback}"
            )
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
        feedback = cast(str, typer.prompt("What should change in the plan?")).strip()
        if not feedback:
            console.print(
                "[yellow]Empty feedback; keeping original goal for next iteration.[/yellow]"
            )
            continue
        current_goal = plan_policy.build_replan_goal(goal, feedback, result)
    console.print(f"[red]Reached max iterations ({max_iterations}) without acceptance.[/red]")
    raise typer.Exit(code=1)


def execute_plan(
    planner: Planner,
    goal: str,
    project_root: Path,
    effective_spec: Path,
    interactive: bool,
    max_iterations: int,
    cfg: MilknadoConfig,
) -> None:
    """Render the planning banner and drive the CLI planning session."""
    console.print(f"[bold]Planning:[/bold] {goal}")
    if interactive:
        _exec_plan_interactive(planner, goal, project_root, effective_spec, max_iterations, cfg)
    else:
        _exec_plan_non_interactive(planner, goal, project_root, effective_spec, cfg)


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


def _fetch_issue(issue_ref: str) -> _FetchedIssue:
    from milknado.adapters.gh import GhTransportError, gh_issue_view

    try:
        issue = gh_issue_view(issue_ref)
        return _FetchedIssue(issue.title, issue.body, issue.number, issue.url)
    except GhTransportError as exc:
        console.print(f"[red]gh issue view {issue_ref} failed:[/red] {exc}")
        raise typer.Exit(code=1) from None


def _resolve_plan_spec(
    spec_paths: list[Path],
    issue_refs: list[str],
    project_root: Path,
) -> Path:
    from milknado.domains.planning import resolve_plan_spec

    return resolve_plan_spec(spec_paths, issue_refs, project_root, _fetch_issue)


def _derive_goal(spec_path: Path) -> str:
    from milknado.domains.planning import derive_goal

    try:
        return derive_goal(spec_path)
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(
            f"--spec file is not valid UTF-8 text: {spec_path} ({exc})"
        ) from None


def plan(
    spec: Annotated[
        list[str] | None,
        typer_option(
            "--spec",
            help=(
                "Spec .md file(s). Repeat the flag or pass comma-separated paths "
                "to combine multiple specs."
            ),
        ),
    ] = None,
    issue: Annotated[
        list[str] | None,
        typer_option(
            "--issue",
            help=(
                "GitHub issue ref (number, owner/repo#123, or URL) fetched via gh. "
                "Repeat the flag or pass comma-separated refs to combine issues."
            ),
        ),
    ] = None,
    project_root: Annotated[
        Path, typer_option("--project-root", help="Project root directory")
    ] = DEFAULT_PROJECT_ROOT,
    interactive: Annotated[
        bool,
        typer_option(
            "--interactive/--no-interactive",
            help="Iterate on planning with accept/revise/cancel prompts.",
        ),
    ] = False,
    max_iterations: Annotated[
        int,
        typer_option(
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
        execute_plan(
            planner, goal, project_root, effective_spec, interactive, max_iterations, config
        )
    finally:
        graph.close()
