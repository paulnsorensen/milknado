"""Application-layer plan policy: subprocess port, critic, exec, interactive."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.config import MilknadoConfig
    from milknado.domains.planning import Planner
    from milknado.domains.planning.planner import PlanResult

import typer
from rich.console import Console

from milknado.domains.common import build_planning_subprocess
from milknado.domains.common.agent_argv import resolve_planning_agent_command
from milknado.domains.planning.ports import PlanningPorts, PlanningProcessResult

console = Console()


class _PlanningSubprocess:
    def run_agent(
        self,
        context_path: Path,
        command: str,
        project_root: Path,
    ) -> PlanningProcessResult:
        if command == "local-planner":
            configured = os.environ.get("MILKNADO_LOCAL_PLANNER")
            if not configured:
                raise ValueError("local-planner requires MILKNADO_LOCAL_PLANNER to name a script")
            planner_path = Path(configured).resolve()
            root = project_root.resolve()
            if not planner_path.is_file() or not planner_path.is_relative_to(root):
                raise ValueError("MILKNADO_LOCAL_PLANNER must be a file within the project root")
            argv = [sys.executable, str(planner_path)]
            options = {
                "input": context_path.read_text(encoding="utf-8"),
                "text": True,
                "stderr": subprocess.PIPE,
            }
        else:
            argv, options = build_planning_subprocess(
                context_path,
                command,
                project_root=project_root,
            )
        options["stdout"] = subprocess.PIPE
        completed = subprocess.run(argv, cwd=project_root, check=False, **options)
        return PlanningProcessResult(
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    def run_validation(
        self,
        command: str,
        payload: dict[str, object],
        project_root: Path,
    ) -> PlanningProcessResult:
        argv = shlex.split(command, posix=True)
        if not argv:
            return PlanningProcessResult(exit_code=1, stderr="empty planning validation command")
        completed = subprocess.run(
            argv,
            cwd=project_root,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        return PlanningProcessResult(
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


@dataclass(frozen=True)
class CriticVerdict:
    approved: bool
    feedback: str


_VERDICT_TAG_RE = re.compile(r"<verdict>\s*(approve|revise)\s*</verdict>", re.IGNORECASE)
_FEEDBACK_TAG_RE = re.compile(r"<feedback>(.*?)</feedback>", re.DOTALL | re.IGNORECASE)

_CRITIC_PROMPT_TEMPLATE = (
    "You are the adversarial plan critic. Review the plan below against the goal.\n\n"
    "## Goal\n\n{goal}\n\n"
    "## Plan manifest\n\n{manifest}\n\n"
    "Respond with exactly one verdict tag, and feedback when revising:\n"
    "<verdict>approve</verdict> or <verdict>revise</verdict>\n"
    "<feedback>free-text feedback explaining what to change</feedback>"
)

_CRITIC_REPROMPT_SUFFIX = (
    "\n\n## Re-prompt\n\n"
    "Your previous response did not include a valid "
    "<verdict>approve</verdict> or <verdict>revise</verdict> tag. "
    "Respond again with exactly one verdict tag."
)


def _parse_critic_output(output: str) -> CriticVerdict | None:
    match = _VERDICT_TAG_RE.search(output)
    if match is None:
        return None
    feedback_match = _FEEDBACK_TAG_RE.search(output)
    feedback = feedback_match.group(1).strip() if feedback_match else ""
    return CriticVerdict(approved=match.group(1).lower() == "approve", feedback=feedback)


def _spawn_plan_critic(command: str, prompt: str, project_root: Path) -> str:
    context_path = project_root / ".milknado" / "plan-critic-context.md"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(prompt, encoding="utf-8")
    argv, options = build_planning_subprocess(context_path, command, project_root=project_root)
    options["stdout"] = subprocess.PIPE
    options.setdefault("stderr", subprocess.PIPE)
    completed = subprocess.run(argv, cwd=project_root, check=False, **options)
    return completed.stdout or ""


def run_plan_critic(goal: str, manifest: str, cfg: MilknadoConfig) -> CriticVerdict:
    """Spawn cfg.plan_reviewer_agent and parse its approve/revise verdict."""
    command = resolve_planning_agent_command(
        cfg.agent_family, planning_agent=cfg.plan_reviewer_agent
    )
    prompt = _CRITIC_PROMPT_TEMPLATE.format(goal=goal, manifest=manifest)
    for _ in range(2):
        output = _spawn_plan_critic(command, prompt, cfg.project_root)
        verdict = _parse_critic_output(output)
        if verdict is not None:
            return verdict
        prompt += _CRITIC_REPROMPT_SUFFIX
    console.print("[red]Plan critic returned an unparseable <verdict> after one re-prompt.[/red]")
    raise typer.Exit(code=1)


def _plan_summary(result: PlanResult) -> str:
    summary = (
        f"Planned {result.change_count} changes \u2192 {result.batch_count} batches"
        f" ({result.oversized_count} oversized), solver={result.solver_status};"
        f" {result.nodes_created} Mikado nodes created"
    )
    if result.mega_batch_change_count is not None:
        summary += f"; mega-batch: {result.mega_batch_change_count} changes in one batch"
    return summary


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


def _run_plan_with_critic(
    planner: Planner,
    goal: str,
    project_root: Path,
    effective_spec: Path,
    cfg: MilknadoConfig,
) -> tuple[PlanResult, CriticVerdict | None]:
    result = planner.launch(goal, project_root, spec_path=effective_spec)
    if not cfg.plan_reviewer_agent:
        return result, None
    verdict: CriticVerdict | None = None
    for _ in range(cfg.plan_review_max_rounds):
        verdict = run_plan_critic(goal, _plan_summary(result), cfg)
        if verdict.approved:
            return result, verdict
        console.print(f"[yellow]Plan critic requested revisions:[/yellow] {verdict.feedback}")
        revised_goal = _build_replan_goal(goal, verdict.feedback, result)
        result = planner.launch(revised_goal, project_root, spec_path=effective_spec)
    return result, verdict


def _exec_plan_non_interactive(
    planner: Planner,
    goal: str,
    project_root: Path,
    effective_spec: Path,
    cfg: MilknadoConfig,
) -> None:
    result, verdict = _run_plan_with_critic(planner, goal, project_root, effective_spec, cfg)
    console.print(_plan_summary(result))
    if verdict is not None and not verdict.approved:
        console.print(
            f"[red]Plan critic did not approve within {cfg.plan_review_max_rounds} "
            f"round(s):[/red] {verdict.feedback}"
        )
        raise typer.Exit(code=1)
    exit_code = _plan_exit_code(result)
    if result.solver_status == "UNKNOWN" and result.batch_count >= 1:
        Console(stderr=True).print(
            "[yellow]Warning: solver returned UNKNOWN \u2014 results may be suboptimal[/yellow]"
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
        result, verdict = _run_plan_with_critic(
            planner, current_goal, project_root, effective_spec, cfg
        )
        console.print(_plan_iteration_summary(iteration, result))
        if result.context_path is not None:
            console.print(f"[dim]Planner context: {result.context_path}[/dim]")
        if verdict is not None and not verdict.approved:
            console.print(
                f"[yellow]Plan critic did not approve within {cfg.plan_review_max_rounds} "
                f"round(s); falling back to manual review:[/yellow] {verdict.feedback}"
            )
        exit_code = _plan_exit_code(result)
        if result.solver_status == "UNKNOWN" and result.batch_count >= 1:
            Console(stderr=True).print(
                "[yellow]Warning: solver returned UNKNOWN"
                " \u2014 results may be suboptimal[/yellow]"
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


def build_planner(graph, project_root: Path, cfg: MilknadoConfig) -> Planner:  # noqa: ANN001
    """Create a Planner with the CRG and tilth adapters wired."""
    from milknado.adapters.crg import CrgAdapter
    from milknado.adapters.tilth import TilthAdapter
    from milknado.domains.planning import Planner

    crg = CrgAdapter(project_root)
    return Planner(
        graph,
        crg,
        cfg.planning_agent,
        PlanningPorts(tilth=TilthAdapter(), process=_PlanningSubprocess()),
        planning_validation_hook=cfg.planning_validation_hook,
        prompt_prepend=cfg.planning_prompt_prepend,
    )


def execute_plan(
    planner: Planner,
    goal: str,
    project_root: Path,
    effective_spec: Path,
    interactive: bool,
    max_iterations: int,
    cfg: MilknadoConfig,
) -> None:
    """Route to interactive or non-interactive plan execution."""
    if not interactive:
        _exec_plan_non_interactive(planner, goal, project_root, effective_spec, cfg)
    else:
        _exec_plan_interactive(planner, goal, project_root, effective_spec, max_iterations, cfg)
