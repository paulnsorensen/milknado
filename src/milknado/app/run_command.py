from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

if TYPE_CHECKING:
    from milknado.domains.common import MilknadoConfig
    from milknado.domains.execution import ExecutionConfig, RunLoopResult
    from milknado.domains.graph import MikadoGraph
    from milknado.domains.planning.planner import PlanResult

console = Console()
CONFIG_FILE = "milknado.toml"


def _ensure_plugins_loaded(config: MilknadoConfig) -> None:
    from milknado.plugins import discover_entry_point_plugins, load_plugins

    for plugin in load_plugins(config.plugins):
        console.print(f"  Plugin loaded: {plugin.meta.name}")
    for plugin in discover_entry_point_plugins():
        console.print(f"  Plugin loaded: {plugin.meta.name} (entry point)")


def _load_or_default(project_root: Path) -> MilknadoConfig:
    from milknado.domains.common import default_config, load_config

    config_path = project_root / CONFIG_FILE
    config = load_config(config_path) if config_path.exists() else default_config(project_root)
    _ensure_plugins_loaded(config)
    return config


def _ensure_db(config: MilknadoConfig) -> MikadoGraph:
    from milknado.domains.graph import MikadoGraph

    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    return MikadoGraph(config.db_path)


def _run_crg(crg: object, project_root: Path, *, rebuild: bool = False) -> None:
    import subprocess

    method = "build_graph" if rebuild else "ensure_graph"
    done_msg = "Code-review-graph rebuilt." if rebuild else "Code-review-graph ready."
    try:
        getattr(crg, method)(project_root)
        console.print(done_msg)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]CRG build failed (exit {e.returncode}).[/red]")
        if e.stderr:
            console.print(e.stderr)
        raise typer.Exit(code=1) from None


def _check_protected_branch(config: MilknadoConfig, branch: str, allow_protected: bool) -> None:
    if not allow_protected and branch in config.protected_branches:
        console.print(
            f"[red]Branch [bold]{branch}[/bold] is protected. "
            "Switch to a feature branch or pass --allow-protected.[/red]"
        )
        raise typer.Exit(code=2)


def _trigger_replan_on_gaps(
    config: MilknadoConfig,
    project_root: Path,
    goal_delta: str,
    spec_path: Path | None,
    max_verify_rounds: int = 3,
) -> None:
    from milknado.adapters.crg import CrgAdapter
    from milknado.domains.graph import MikadoGraph
    from milknado.domains.planning import Planner

    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    graph = MikadoGraph(config.db_path)
    try:
        crg = CrgAdapter(project_root)
        planner = Planner(graph, crg, config.planning_agent)
        console.print(f"[yellow]Gaps detected — replanning: {goal_delta[:80]}[/yellow]")
        planner.launch(
            goal=goal_delta,
            project_root=project_root,
            spec_path=spec_path,
            max_verify_rounds=max_verify_rounds,
        )
    finally:
        graph.close()


def _build_exec_config(
    config: MilknadoConfig,
    project_root: Path,
) -> ExecutionConfig:
    from milknado.domains.execution import ExecutionConfig

    return ExecutionConfig(
        execution_agent=config.execution_agent,
        quality_gates=config.quality_gates,
        worktree_pattern=config.worktree_pattern,
        project_root=project_root,
        dispatch_max_retries=config.dispatch_max_retries,
        dispatch_backoff_seconds=config.dispatch_backoff_seconds,
    )


def _print_run_result(result: RunLoopResult) -> None:
    if result.root_done:
        console.print("[green]All nodes complete. Root goal achieved.[/green]")
    else:
        console.print(
            f"[yellow]Loop ended: {result.completed_total} completed, "
            f"{result.failed_total} failed.[/yellow]"
        )

    for conflict in result.rebase_conflicts:
        console.print(
            f"\n[red bold]Rebase conflict — node {conflict.node_id}:[/red bold] "
            f"{conflict.description}",
        )
        if conflict.conflicting_files:
            for f in conflict.conflicting_files:
                console.print(f"  [red]•[/red] {f}")
        if conflict.detail:
            console.print(f"  [dim]{conflict.detail}[/dim]")


def plan_summary(result: PlanResult, max_verify_rounds: int = 3) -> str:
    cap_marker = " CAP-HIT" if result.verify_round_cap_hit else ""
    coverage = (
        f"verify_rounds={result.verify_rounds_used}/{max_verify_rounds}{cap_marker}"
        f" ({result.coverage_gaps_remaining} gaps)"
    )
    return (
        f"Planned {result.change_count} changes → {result.batch_count} batches"
        f" ({result.oversized_count} oversized), solver={result.solver_status};"
        f" {result.nodes_created} Mikado nodes created; {coverage}"
    )


def plan_exit_code(result: PlanResult) -> int:
    if result.solver_status in ("INFEASIBLE", "NO_MANIFEST"):
        return 1
    if result.solver_status in ("OPTIMAL", "FEASIBLE"):
        return 0
    if result.solver_status == "UNKNOWN" and result.batch_count >= 1:
        return 0
    if not result.success:
        return result.exit_code
    return 0


def execute_plan(
    project_root: Path,
    spec_paths: list[Path],
    issue_refs: list[str],
    max_verify_rounds: int,
    resume: bool,
    reset: bool,
    mega_batch_threshold: int,
    force_single_batch: bool,
) -> None:
    from milknado.adapters.crg import CrgAdapter
    from milknado.app.spec_ingest import (
        SpecIngestError,
        derive_goal,
        materialize_combined_spec,
        materialize_issue_spec,
    )
    from milknado.domains.common.errors import ExistingPlanDetected
    from milknado.domains.planning import Planner

    config = _load_or_default(project_root)
    graph = _ensure_db(config)
    try:
        if len(spec_paths) == 1 and not issue_refs:
            effective_spec = spec_paths[0]
        elif not spec_paths and issue_refs:
            effective_spec = materialize_issue_spec(issue_refs, project_root)
        else:
            effective_spec = materialize_combined_spec(spec_paths, issue_refs, project_root)
    except SpecIngestError as e:
        graph.close()
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None

    goal = derive_goal(effective_spec)
    crg = CrgAdapter(project_root)
    planner = Planner(graph, crg, config.planning_agent)
    console.print(f"[bold]Planning:[/bold] {goal}")
    try:
        result = planner.launch(
            goal,
            project_root,
            spec_path=effective_spec,
            max_verify_rounds=max_verify_rounds,
            resuming=resume,
            reset=reset,
            mega_batch_threshold=mega_batch_threshold,
            force_single_batch=force_single_batch,
        )
    except ExistingPlanDetected as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    finally:
        graph.close()

    console.print(plan_summary(result, max_verify_rounds))
    code = plan_exit_code(result)
    if result.solver_status == "UNKNOWN" and result.batch_count >= 1:
        Console(stderr=True).print(
            "[yellow]Warning: solver returned UNKNOWN — results may be suboptimal[/yellow]"
        )
    if code != 0:
        raise typer.Exit(code=code)


def execute_run(
    project_root: Path,
    strict: bool,
    allow_protected: bool,
    spec: Path | None,
    max_verify_rounds: int,
) -> None:
    from milknado.adapters import CrgAdapter, GitAdapter, RalphifyAdapter
    from milknado.domains.execution import Executor, RunLoop, get_dispatchable_nodes

    config = _load_or_default(project_root)
    graph = _ensure_db(config)
    try:
        if not get_dispatchable_nodes(graph):
            console.print("No nodes ready for execution.")
            return

        git = GitAdapter(project_root)
        feature_branch = git.current_branch()
        _check_protected_branch(config, feature_branch, allow_protected)

        spec_text = spec.read_text(encoding="utf-8") if spec else None
        ralph = RalphifyAdapter()
        crg = CrgAdapter(project_root)
        executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph, config=config)
        console.print(f"Starting execution loop on [bold]{feature_branch}[/bold]...")
        result = loop.run(
            config=_build_exec_config(config, project_root),
            feature_branch=feature_branch,
            concurrency_limit=config.concurrency_limit,
            strict=strict,
            spec_text=spec_text,
            spec_path=spec,
        )
    finally:
        graph.close()

    _print_run_result(result)

    if (
        result.verify_outcome is not None
        and not result.verify_outcome.done
        and result.verify_outcome.goal_delta
    ):
        _trigger_replan_on_gaps(
            config,
            project_root,
            result.verify_outcome.goal_delta,
            spec,
            max_verify_rounds=max_verify_rounds,
        )

    if result.failed_total > 0 or result.strict_exit:
        raise typer.Exit(code=1)
