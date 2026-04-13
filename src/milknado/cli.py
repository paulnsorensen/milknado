from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from milknado.domains.common import (
    MilknadoConfig,
    default_config,
    load_config,
    save_config,
)
from milknado.domains.graph import MikadoGraph, render_tree

app = typer.Typer(name="milknado", help="Mikado execution engine")
console = Console()

CONFIG_FILE = "milknado.toml"


def _find_config(project_root: Path) -> Path:
    return project_root / CONFIG_FILE


def _load_or_default(project_root: Path) -> MilknadoConfig:
    config_path = _find_config(project_root)
    if config_path.exists():
        return load_config(config_path)
    return default_config(project_root)


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

    graph = _ensure_db(config)
    graph.close()
    console.print(f"Database ready: {config.db_path}")

    crg = CrgAdapter(project_root)
    crg.ensure_graph(project_root)
    console.print("Code-review-graph ready.")


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

        output = render_tree(graph)
        console.print(output)
    finally:
        graph.close()


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

        if parent is not None:
            parent_node = graph.get_node(parent)
            if parent_node and parent_node.status.value == "running":
                graph.mark_blocked(parent)
                console.print(f"Parent node {parent} marked as blocked.")

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


@app.command()
def run(
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Execute ready leaf nodes as parallel ralph loops."""
    from milknado.adapters import CrgAdapter, GitAdapter, RalphifyAdapter
    from milknado.domains.execution import (
        ExecutionConfig,
        Executor,
        RunLoop,
        get_dispatchable_nodes,
    )

    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        dispatchable = get_dispatchable_nodes(graph)
        if not dispatchable:
            console.print("No nodes ready for execution.")
            return

        git = GitAdapter(project_root)
        ralph = RalphifyAdapter()
        crg = CrgAdapter(project_root)
        executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)

        exec_config = ExecutionConfig(
            agent_command=config.agent_command,
            quality_gates=config.quality_gates,
            worktree_pattern=config.worktree_pattern,
            project_root=project_root,
        )

        feature_branch = git.current_branch()
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        console.print(f"Starting execution loop on [bold]{feature_branch}[/bold]...")
        result = loop.run(
            config=exec_config,
            feature_branch=feature_branch,
            concurrency_limit=config.concurrency_limit,
        )

        if result.root_done:
            console.print("[green]All nodes complete. Root goal achieved.[/green]")
        else:
            console.print(
                f"[yellow]Loop ended: {result.completed_total} completed, "
                f"{result.failed_total} failed.[/yellow]"
            )
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
