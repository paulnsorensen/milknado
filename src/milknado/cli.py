from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from milknado.domains.execution import ExecutionConfig
    from milknado.domains.execution.run_loop import RunLoopResult

import typer
from rich.console import Console

from milknado.domains.common import (
    MikadoNode,
    MilknadoConfig,
    NodeStatus,
    default_config,
    load_config,
    save_config,
)
from milknado.domains.graph import MikadoGraph, render_tree

app = typer.Typer(name="milknado", help="Mikado execution engine")
console = Console()

CONFIG_FILE = "milknado.toml"

def _ensure_plugins_loaded(config: MilknadoConfig) -> None:
    from milknado.plugins import discover_entry_point_plugins, load_plugins

    for plugin in load_plugins(config.plugins):
        console.print(f"  Plugin loaded: {plugin.meta.name}")
    for plugin in discover_entry_point_plugins():
        console.print(f"  Plugin loaded: {plugin.meta.name} (entry point)")


def _maybe_block_parent(graph: MikadoGraph, parent: int | None) -> None:
    if parent is None:
        return
    parent_node = graph.get_node(parent)
    if parent_node and parent_node.status.value == "running":
        graph.mark_blocked(parent)
        console.print(f"Parent node {parent} marked as blocked.")


def _find_config(project_root: Path) -> Path:
    return project_root / CONFIG_FILE


def _load_or_default(project_root: Path) -> MilknadoConfig:
    config_path = _find_config(project_root)
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = default_config(project_root)
    _ensure_plugins_loaded(config)
    return config


def _ensure_db(config: MilknadoConfig) -> MikadoGraph:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    return MikadoGraph(config.db_path)


@app.command()
def init(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
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
        save_config(config, config_path)
        console.print(f"Created config: {config_path}")

    _ensure_plugins_loaded(config)
    graph = _ensure_db(config)
    graph.close()
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


@app.command()
def index(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
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


@app.command()
def status(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
) -> None:
    """Show the current state of the Mikado graph."""
    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        nodes = graph.get_all_nodes()
        if not nodes:
            console.print("No nodes in graph. Run [bold]milknado plan[/bold] to start.")
            return

        run_states = _fetch_run_states(nodes)
        output = render_tree(graph, run_states=run_states)
        console.print(output)
    finally:
        graph.close()


def _fetch_run_states(nodes: list[MikadoNode]) -> dict[str, str] | None:
    run_ids = [n.run_id for n in nodes if n.run_id and n.status == NodeStatus.RUNNING]
    if not run_ids:
        return None
    try:
        from milknado.adapters.ralphify import RalphifyAdapter

        ralph = RalphifyAdapter()
        states: dict[str, str] = {}
        for run_id in run_ids:
            run = ralph.get_run(run_id)
            if run:
                states[run_id] = getattr(run, "status", "unknown")
        return states or None
    except Exception:
        return None


@app.command()
def crg(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
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
    parent: Annotated[
        int | None, typer.Option("--parent", "-p", help="Parent node ID")
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
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        node = graph.add_node(description, parent_id=parent)
        if files:
            graph.set_file_ownership(node.id, files)

        _maybe_block_parent(graph, parent)
        console.print(f"Added node {node.id}: {node.description}")
    finally:
        graph.close()


@app.command()
def plan(
    goal: Annotated[str, typer.Argument(help="Goal to decompose")],
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Launch interactive planning session to decompose a goal."""
    from milknado.adapters.crg import CrgAdapter
    from milknado.domains.planning import Planner

    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        crg = CrgAdapter(project_root)
        planner = Planner(graph, crg, config.agent_command)
        console.print(f"[bold]Planning:[/bold] {goal}")
        result = planner.launch(goal, project_root)
        if result.success:
            console.print("[green]Planning session complete.[/green]")
        else:
            console.print(
                f"[red]Planning session exited with code {result.exit_code}.[/red]"
            )
            raise typer.Exit(code=result.exit_code)
    finally:
        graph.close()


def _build_exec_config(
    config: MilknadoConfig, project_root: Path,
) -> ExecutionConfig:
    from milknado.domains.execution import ExecutionConfig

    return ExecutionConfig(
        agent_command=config.agent_command,
        quality_gates=config.quality_gates,
        worktree_pattern=config.worktree_pattern,
        project_root=project_root,
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


@app.command()
def run(
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Execute ready leaf nodes as parallel ralph loops."""
    from milknado.adapters import CrgAdapter, GitAdapter, RalphifyAdapter
    from milknado.domains.execution import Executor, RunLoop, get_dispatchable_nodes

    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        if not get_dispatchable_nodes(graph):
            console.print("No nodes ready for execution.")
            return

        git = GitAdapter(project_root)
        ralph = RalphifyAdapter()
        crg = CrgAdapter(project_root)
        executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)

        feature_branch = git.current_branch()
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        console.print(f"Starting execution loop on [bold]{feature_branch}[/bold]...")
        result = loop.run(
            config=_build_exec_config(config, project_root),
            feature_branch=feature_branch,
            concurrency_limit=config.concurrency_limit,
        )
        _print_run_result(result)
    finally:
        graph.close()


plugin_app = typer.Typer(name="plugin", help="Plugin management commands")
app.add_typer(plugin_app)


@plugin_app.command("init")
def plugin_init(
    name: Annotated[str, typer.Argument(help="Plugin name")],
    target_dir: Annotated[
        Path, typer.Option("--target-dir", "-d", help="Directory to create plugin in")
    ] = Path("."),
) -> None:
    """Scaffold a new milknado plugin."""
    from milknado.plugins import scaffold_plugin

    try:
        result = scaffold_plugin(name, target_dir.resolve())
        console.print(f"Created plugin [bold]{name}[/bold] at {result.plugin_dir}")
        for f in result.files_created:
            console.print(f"  {f}")
    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
