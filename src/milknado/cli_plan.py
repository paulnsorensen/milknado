"""plan command and supporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from milknado.domains.planning import Planner
    from milknado.domains.planning.planner import PlanResult

import typer
from rich.console import Console

from milknado._cli_helpers import _ensure_db, _load_or_default
from milknado.cli_plan_specs import _resolve_plan_spec

console = Console()


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


def _resolve_plan_inputs(
    spec: list[str] | None, issue: list[str] | None, project_root: Path
) -> Path:
    """Validate --spec/--issue inputs and materialize the effective spec path."""
    spec_paths = _validate_spec_paths(_flatten_csv(spec))
    issue_refs = _flatten_csv(issue)
    if not spec_paths and not issue_refs:
        console.print("[red]Provide --spec or --issue.[/red]")
        raise typer.Exit(code=1)
    return _resolve_plan_spec(spec_paths, issue_refs, project_root)


def _run_planner(
    graph,
    config,
    project_root: Path,
    goal: str,
    effective_spec: Path,
    interactive: bool,
    max_iterations: int,
) -> None:
    """Construct the planner and dispatch the interactive or one-shot run."""
    from milknado.adapters.crg import CrgAdapter
    from milknado.domains.planning import Planner

    crg = CrgAdapter(project_root)
    planner = Planner(
        graph,
        crg,
        config.planning_agent,
        planning_validation_hook=config.planning_validation_hook,
        prompt_prepend=config.planning_prompt_prepend,
    )
    console.print(f"[bold]Planning:[/bold] {goal}")
    if not interactive:
        _exec_plan_non_interactive(planner, goal, project_root, effective_spec)
    else:
        _exec_plan_interactive(planner, goal, project_root, effective_spec, max_iterations)


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
    project_root = project_root.resolve()
    effective_spec = _resolve_plan_inputs(spec, issue, project_root)

    goal = _derive_goal(effective_spec)
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)

    try:
        _run_planner(
            graph, config, project_root, goal, effective_spec, interactive, max_iterations
        )
    finally:
        graph.close()
