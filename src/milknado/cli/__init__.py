"""Milknado CLI — facade that wires all sub-command modules into one app."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated, cast

import typer

from milknado.cli._helpers import (
    DEFAULT_PROJECT_ROOT,
    console,
    find_config,
    load_or_default,
    maybe_block_parent,
    open_project_for_cli,
    typer_argument,
    typer_option,
)
from milknado.cli.agents import agents_app
from milknado.cli.config import config_app
from milknado.cli.github_roadmap import github_roadmap_app
from milknado.cli.graph import edge_app, graph_app
from milknado.cli.plan import _derive_goal as _derive_goal  # re-export
from milknado.cli.plan import plan
from milknado.cli.roadmap import roadmap_app
from milknado.cli.run import attach, run, watch
from milknado.cli.tools import (
    install_rust_tools_or_exit,
    plugin_app,
    tools_app,
    write_worker_hooks,
)
from milknado.domains.common import (
    MikadoNode,
    NodeKind,
    NodeSpec,
    NodeStatus,
    default_config,
    detect_project_gates,
    load_config,
    normalize_hint_paths,
    save_config,
    validate_hint_path,
)
from milknado.domains.graph import render_tree

ProjectRootArgument = Annotated[Path, typer_argument(help="Project root directory")]
ProjectRootOption = Annotated[Path, typer_option("--project-root", help="Project root directory")]
__all__ = ["_derive_goal", "app"]

app = typer.Typer(name="milknado", help="Mikado execution engine")

_ = app.add_typer(agents_app)
_ = app.add_typer(config_app)
_ = app.add_typer(edge_app)
_ = app.add_typer(graph_app)
_ = app.add_typer(plugin_app)
_ = app.add_typer(tools_app)
_ = app.add_typer(roadmap_app)
_ = app.add_typer(github_roadmap_app)
_ = app.command()(plan)
_ = app.command()(run)
_ = app.command()(attach)
_ = app.command()(watch)


@app.command()
def init(
    project_root: ProjectRootArgument = DEFAULT_PROJECT_ROOT,
    install_rust_tools: Annotated[
        bool,
        typer_option(
            "--install-rust-tools",
            help="Install missing mergiraf/rtk via cargo.",
        ),
    ] = False,
) -> None:
    """Initialize milknado in a project directory."""
    from milknado.adapters.crg import CrgAdapter

    project_root = project_root.resolve()
    config_path = find_config(project_root)

    if config_path.exists():
        console.print(f"Config already exists: {config_path}")
        config = load_config(config_path)
    else:
        config = default_config(project_root)
        detected_gates = detect_project_gates(project_root)
        if detected_gates is not None:
            config = config.__replace__(quality_gates=detected_gates)
            gate_cmds = ", ".join(f"`{g.command}`" for g in detected_gates)
            console.print(f"Detected project type → gates: {gate_cmds}")
        else:
            console.print(
                "[yellow]No project type detected — quality_gates not set.[/yellow]\n"
                + "[yellow]milknado fails closed until quality_gates is configured.[/yellow]\n"
                + "[yellow]Add [milknado] quality_gates to milknado.toml, "
                + "or run /milknado-config to set up per-flavor gates.[/yellow]"
            )
        save_config(config, config_path)
        console.print(f"Created config: {config_path}")

    project = open_project_for_cli(project_root, config)
    project.graph.close()
    console.print(f"Database ready: {config.db_path}")

    crg = CrgAdapter(project_root)
    try:
        crg.ensure_graph(project_root)
        console.print("Code-review-graph ready.")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]CRG build failed (exit {e.returncode}).[/red]")
        if stderr := cast(str | bytes | None, e.stderr):
            console.print(stderr)
        raise typer.Exit(code=1) from None

    if install_rust_tools:
        install_rust_tools_or_exit()

    write_worker_hooks(project_root, config)


@app.command()
def index(
    project_root: ProjectRootArgument = DEFAULT_PROJECT_ROOT,
) -> None:
    """Rebuild the code-review-graph index."""
    from milknado.adapters.crg import CrgAdapter

    project_root = project_root.resolve()
    _ = load_or_default(project_root)
    crg = CrgAdapter(project_root)
    try:
        crg.build_graph(project_root)
        console.print("Code-review-graph rebuilt.")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]CRG build failed (exit {e.returncode}).[/red]")
        if stderr := cast(str | bytes | None, e.stderr):
            console.print(stderr)
        raise typer.Exit(code=1) from None


def _fetch_run_states(
    nodes: list[MikadoNode],
) -> tuple[dict[str, str] | None, list[str]]:
    run_ids = [n.run_id for n in nodes if n.run_id and n.status == NodeStatus.RUNNING]
    if not run_ids:
        return None, []
    from milknado.adapters.loop import LoopAdapter

    adapter = LoopAdapter()
    states: dict[str, str] = {}
    failures: list[str] = []
    for run_id in run_ids:
        try:
            run = adapter.get_run(run_id)
        except Exception as exc:
            failures.append(f"{run_id}: {type(exc).__name__}: {exc}")
            continue
        if run:
            states[run_id] = getattr(run, "status", "unknown")
    return states or None, failures


@app.command()
def status(
    project_root: ProjectRootArgument = DEFAULT_PROJECT_ROOT,
    show_all: Annotated[
        bool,
        typer_option("--all", help="Include archived (soft-hidden) nodes."),
    ] = False,
) -> None:
    """Show the current state of the Mikado graph."""
    project_root = project_root.resolve()
    project = open_project_for_cli(project_root)
    graph = project.graph

    try:
        nodes = graph.get_all_nodes(include_archived=show_all)
        if not nodes:
            message = "No nodes in graph. Run [bold]milknado plan[/bold] to start."
            if not show_all:
                archived = graph.count_archived()
                if archived:
                    message += f" ({archived} archived node(s) — use --all)"
            console.print(message)
            return

        run_states, failures = _fetch_run_states(nodes)
        if failures:
            console.print(
                f"[yellow]Degraded status: {len(failures)} run state"
                + f"{'s' if len(failures) != 1 else ''} unavailable "
                + f"({'; '.join(failures)})[/yellow]"
            )
        console.print(render_tree(graph, run_states=run_states, include_archived=show_all))
    finally:
        graph.close()


@app.command()
def rebalance(
    project_root: ProjectRootArgument = DEFAULT_PROJECT_ROOT,
    dry_run: Annotated[
        bool,
        typer_option("--dry-run", help="Print the plan without mutating anything."),
    ] = False,
    no_sweep: Annotated[
        bool,
        typer_option("--no-sweep", help="Skip archiving all-DONE subtrees."),
    ] = False,
    no_restructure: Annotated[
        bool,
        typer_option("--no-restructure", help="Skip orphan grouping and structural findings."),
    ] = False,
    no_reap: Annotated[
        bool,
        typer_option("--no-reap", help="Skip archived-worktree teardown."),
    ] = False,
) -> None:
    """Rebalance the working tree: sweep finished work, regroup orphans, reap worktrees."""
    from milknado.app.rebalance import RebalanceOptions
    from milknado.app.rebalance import rebalance as run_rebalance
    from milknado.domains.graph import render_report

    options = RebalanceOptions(
        dry_run=dry_run,
        sweep=not no_sweep,
        restructure=not no_restructure,
        reap=not no_reap,
    )
    report = run_rebalance(project_root.resolve(), options)
    console.print(render_report(report, dry_run=dry_run), markup=False)


@app.command()
def crg(
    project_root: ProjectRootArgument = DEFAULT_PROJECT_ROOT,
) -> None:
    """Show the code-review-graph architecture overview."""
    import json

    from milknado.adapters.crg import CrgAdapter

    project_root = project_root.resolve()
    if not (project_root / ".code-review-graph" / "graph.db").exists():
        console.print("[dim]No CRG index. Run [bold]milknado index[/bold] to build.[/dim]")
        raise typer.Exit(code=1)

    adapter = CrgAdapter(project_root)
    overview = adapter.get_architecture_overview()
    formatted = json.dumps(overview, indent=2, default=str)
    console.print(formatted)


@app.command("add-node")
def add_node(
    description: Annotated[str, typer_argument(help="Node description")],
    parent: Annotated[int | None, typer_option("--parent", "-p", help="Parent node ID")] = None,
    kind: Annotated[
        NodeKind,
        typer_option("--kind", "-k", help="Node kind"),
    ] = NodeKind.TASK,
    flavor: Annotated[
        str | None,
        typer_option("--flavor", help="Task flavor, including config-registered names"),
    ] = None,
    artifact: Annotated[
        str | None,
        typer_option("--artifact", help="Artifact path associated with this node"),
    ] = None,
    prereq: Annotated[
        list[int] | None,
        typer_option("--prereq", help="Prerequisite node ID; repeatable"),
    ] = None,
    files: Annotated[
        list[str] | None,
        typer_option("--files", "-f", help="Files this node will touch"),
    ] = None,
    project_root: ProjectRootOption = DEFAULT_PROJECT_ROOT,
) -> None:
    """Add a node to the Mikado graph."""
    project_root = project_root.resolve()
    project = open_project_for_cli(project_root)
    graph = project.graph

    try:
        if artifact is not None:
            validate_hint_path(artifact, project_root, label="artifact")
        node = graph.add_node(
            description,
            parent_id=parent,
            spec=NodeSpec(
                kind=kind,
                flavor=flavor,
                artifact_path=artifact,
                prereqs=tuple(prereq or ()),
                flavor_registry=project.config.flavor_registry,
            ),
        )
        if files:
            graph.set_file_ownership(node.id, normalize_hint_paths(files, project_root))
        maybe_block_parent(graph, parent)
        console.print(f"Added node {node.id}: {node.description}")
    finally:
        graph.close()


@app.command()
def doctor(
    project_root: ProjectRootArgument = DEFAULT_PROJECT_ROOT,
) -> None:
    """Run health checks on the milknado installation."""
    from milknado.domains.common import render_report, run_doctor

    project_root = project_root.resolve()
    config_path = find_config(project_root)
    config, _ = load_or_default(project_root)
    report = run_doctor(config_path, config)
    text, issue_count = render_report(report)
    typer.echo(text)
    if issue_count > 0:
        raise typer.Exit(code=1)
