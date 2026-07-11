"""Milknado CLI — facade that wires all sub-command modules into one app."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from milknado._cli_helpers import (
    _ensure_db,
    _ensure_plugins_loaded,
    _find_config,
    _load_or_default,
    _maybe_block_parent,
    console,
)
from milknado.cli_agents import agents_app
from milknado.cli_github_roadmap import github_roadmap_app
from milknado.cli_graph import edge_app
from milknado.cli_plan import _derive_goal as _derive_goal  # noqa: PLC0414 — re-export
from milknado.cli_plan import plan
from milknado.cli_roadmap import roadmap_app
from milknado.cli_run import attach, run
from milknado.cli_tools import (
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
    load_config,
    save_config,
)
from milknado.domains.common.config import detect_project_gates
from milknado.domains.common.paths import normalize_hint_paths
from milknado.domains.graph import render_tree

# Public surface of this facade: the Typer app plus the `_derive_goal`
# re-export consumed by tests/adversarial/test_cli_chaos.py. Listing the
# re-export here marks it exported (clears CodeQL py/unused-import #55).
__all__ = ["_derive_goal", "app"]

app = typer.Typer(name="milknado", help="Mikado execution engine")

app.add_typer(agents_app)
app.add_typer(edge_app)
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

    plugins = _ensure_plugins_loaded(config)
    graph = _ensure_db(config, plugins)
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


def _fetch_run_states(nodes: list[MikadoNode]) -> dict[str, str] | None:
    run_ids = [n.run_id for n in nodes if n.run_id and n.status == NodeStatus.RUNNING]
    if not run_ids:
        return None
    try:
        from milknado.adapters.loop import LoopAdapter

        ralph = LoopAdapter()
        states: dict[str, str] = {}
        for run_id in run_ids:
            run = ralph.get_run(run_id)
            if run:
                states[run_id] = getattr(run, "status", "unknown")
        return states or None
    except Exception:  # noqa: BLE001 — status enrichment is best-effort; never block render
        return None


@app.command()
def status(
    project_root: Annotated[Path, typer.Argument(help="Project root directory")] = Path("."),
) -> None:
    """Show the current state of the Mikado graph."""
    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)

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
        str,
        typer.Option("--kind", "-k", help="Node kind: roadmap|goal|task"),
    ] = "task",
    files: Annotated[
        list[str] | None,
        typer.Option("--files", "-f", help="Files this node will touch"),
    ] = None,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Add a node to the Mikado graph."""
    try:
        node_kind = NodeKind(kind)
    except ValueError:
        valid = sorted(k.value for k in NodeKind)
        raise typer.BadParameter(f"invalid kind {kind!r}; expected one of {valid}") from None
    project_root = project_root.resolve()
    config, plugins = _load_or_default(project_root)
    graph = _ensure_db(config, plugins)

    try:
        node = graph.add_node(description, parent_id=parent, spec=NodeSpec(kind=node_kind))
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
    from milknado.domains.common.doctor import render_report, run_doctor

    project_root = project_root.resolve()
    config_path = _find_config(project_root)
    config, _ = _load_or_default(project_root)
    report = run_doctor(config_path, config)
    text, issue_count = render_report(report)
    typer.echo(text)
    if issue_count > 0:
        raise typer.Exit(code=1)
