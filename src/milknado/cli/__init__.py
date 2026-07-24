"""Milknado CLI — facade that wires all sub-command modules into one app."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from milknado.cli._helpers import (
    _find_config,
    _load_or_default,
    _maybe_block_parent,
    _open_project,
    console,
)
from milknado.cli.agents import agents_app
from milknado.cli.github_roadmap import github_roadmap_app
from milknado.cli.graph import edge_app, graph_app
from milknado.cli.plan import _derive_goal as _derive_goal  # noqa: PLC0414 — re-export
from milknado.cli.plan import plan
from milknado.cli.roadmap import roadmap_app
from milknado.cli.run import attach, run
from milknado.cli.tools import (
    _install_rust_tools_or_exit,
    _write_worker_hooks,
    plugin_app,
    tools_app,
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

# Public surface of this facade: the Typer app plus the `_derive_goal`
# re-export consumed by tests/adversarial/test_cli_chaos.py. Listing the
# re-export here marks it exported (clears CodeQL py/unused-import #55).
__all__ = ["_derive_goal", "app"]

app = typer.Typer(name="milknado", help="Mikado execution engine")

app.add_typer(agents_app)
app.add_typer(edge_app)
app.add_typer(graph_app)
app.add_typer(plugin_app)
app.add_typer(tools_app)
app.add_typer(roadmap_app)
app.add_typer(github_roadmap_app)
app.command()(plan)
app.command()(run)
app.command()(attach)


@app.command()
def init(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
    install_rust_tools: Annotated[
        bool,
        typer.Option(
            "--install-rust-tools",
            help="Install missing tilth/mergiraf/rtk via cargo.",
        ),
    ] = False,
) -> None:
    """Initialize milknado in a project directory."""
    from milknado.adapters.crg import CrgAdapter

    project_root = project_root.resolve()
    config_path = _find_config(project_root)

    if config_path.exists():
        console.print(f"Config already exists: {config_path}")
        config = load_config(config_path)
    else:
        config = default_config(project_root)
        detected_gates = detect_project_gates(project_root)
        if detected_gates is not None:
            config = dataclasses.replace(config, quality_gates=detected_gates)
            gate_cmds = ", ".join(f"`{g.command}`" for g in detected_gates)
            console.print(f"Detected project type → gates: {gate_cmds}")
        else:
            console.print(
                "[yellow]No project type detected — quality_gates not set.[/yellow]\n"
                "[yellow]milknado fails closed until quality_gates is configured.[/yellow]\n"
                "[yellow]Add [milknado] quality_gates to milknado.toml, "
                "or run /milknado-config to set up per-flavor gates.[/yellow]"
            )
        save_config(config, config_path)
        console.print(f"Created config: {config_path}")

    project = _open_project(project_root, config)
    project.graph.close()
    console.print(f"Database ready: {config.db_path}")

    crg = CrgAdapter(project_root)
    try:
        crg.ensure_graph(project_root)
        console.print("Code-review-graph ready.")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]CRG build failed (exit {e.returncode}).[/red]")
        if e.stderr:
            console.print(e.stderr)
        raise typer.Exit(code=1) from None

    if install_rust_tools:
        _install_rust_tools_or_exit()

    _write_worker_hooks(project_root, config)


@app.command()
def index(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
) -> None:
    """Rebuild the code-review-graph index."""
    from milknado.adapters.crg import CrgAdapter

    project_root = project_root.resolve()
    _load_or_default(project_root)
    crg = CrgAdapter(project_root)
    try:
        crg.build_graph(project_root)
        console.print("Code-review-graph rebuilt.")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]CRG build failed (exit {e.returncode}).[/red]")
        if e.stderr:
            console.print(e.stderr)
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
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
    show_all: Annotated[
        bool,
        typer.Option("--all", help="Include archived (soft-hidden) nodes."),
    ] = False,
) -> None:
    """Show the current state of the Mikado graph."""
    project_root = project_root.resolve()
    project = _open_project(project_root)
    graph = project.graph

    try:
        nodes = graph.get_all_nodes(include_archived=show_all)
        if not nodes:
            message = "No nodes in graph. Run [bold]milknado plan[/bold] to start."
            if not show_all:
                archived = sum(
                    1 for n in graph.get_all_nodes(include_archived=True) if n.archived_at
                )
                if archived:
                    message += f" ({archived} archived node(s) — use --all)"
            console.print(message)
            return

        run_states, failures = _fetch_run_states(nodes)
        if failures:
            console.print(
                f"[yellow]Degraded status: {len(failures)} run state"
                f"{'s' if len(failures) != 1 else ''} unavailable "
                f"({'; '.join(failures)})[/yellow]"
            )
        console.print(render_tree(graph, run_states=run_states, include_archived=show_all))
    finally:
        graph.close()


@app.command()
def rebalance(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the plan without mutating anything."),
    ] = False,
    no_sweep: Annotated[
        bool,
        typer.Option("--no-sweep", help="Skip archiving all-DONE subtrees."),
    ] = False,
    no_restructure: Annotated[
        bool,
        typer.Option("--no-restructure", help="Skip orphan grouping and structural findings."),
    ] = False,
    no_reap: Annotated[
        bool,
        typer.Option("--no-reap", help="Skip archived-worktree teardown."),
    ] = False,
) -> None:
    """Rebalance the working tree: sweep finished work, regroup orphans, reap worktrees."""
    from milknado.app.rebalance import rebalance as run_rebalance
    from milknado.domains.graph import render_report

    report = run_rebalance(
        project_root.resolve(),
        dry_run=dry_run,
        sweep=not no_sweep,
        restructure=not no_restructure,
        reap=not no_reap,
    )
    # markup=False: the report's literal "[dry-run]" prefix is not rich markup.
    console.print(render_report(report, dry_run=dry_run), markup=False)


@app.command()
def crg(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
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
    description: Annotated[str, typer.Argument(help="Node description")],
    parent: Annotated[int | None, typer.Option("--parent", "-p", help="Parent node ID")] = None,
    kind: Annotated[
        NodeKind,
        typer.Option("--kind", "-k", help="Node kind"),
    ] = NodeKind.TASK,
    flavor: Annotated[
        str | None,
        typer.Option("--flavor", help="Task flavor, including config-registered names"),
    ] = None,
    artifact: Annotated[
        str | None,
        typer.Option("--artifact", help="Artifact path associated with this node"),
    ] = None,
    prereq: Annotated[
        list[int] | None,
        typer.Option("--prereq", help="Prerequisite node ID; repeatable"),
    ] = None,
    files: Annotated[
        list[str] | None,
        typer.Option("--files", "-f", help="Files this node will touch"),
    ] = None,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Add a node to the Mikado graph."""
    project_root = project_root.resolve()
    project = _open_project(project_root)
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
        _maybe_block_parent(graph, parent)
        console.print(f"Added node {node.id}: {node.description}")
    finally:
        graph.close()


@app.command()
def doctor(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
) -> None:
    """Run health checks on the milknado installation."""
    from milknado.domains.common import render_report, run_doctor

    project_root = project_root.resolve()
    config_path = _find_config(project_root)
    config, _ = _load_or_default(project_root)
    report = run_doctor(config_path, config)
    text, issue_count = render_report(report)
    typer.echo(text)
    if issue_count > 0:
        raise typer.Exit(code=1)
