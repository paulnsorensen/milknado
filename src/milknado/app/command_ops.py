"""Execute functions for simple CLI commands (init, index, status, crg, add-node, doctor)."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from milknado.app._status_display import check_agents_config, print_status
from milknado.app.run_command import (
    CONFIG_FILE,
    _ensure_db,
    _load_or_default,
    _run_crg,
)

console = Console()


def execute_init(project_root: Path) -> None:
    from milknado.adapters.crg import CrgAdapter
    from milknado.domains.common import default_config, load_config, save_config
    from milknado.plugins import discover_entry_point_plugins, load_plugins

    config_path = project_root / CONFIG_FILE
    if config_path.exists():
        console.print(f"Config already exists: {config_path}")
        config = load_config(config_path)
    else:
        config = default_config(project_root)
        save_config(config, config_path)
        console.print(f"Created config: {config_path}")

    for plugin in load_plugins(config.plugins):
        console.print(f"  Plugin loaded: {plugin.meta.name}")
    for plugin in discover_entry_point_plugins():
        console.print(f"  Plugin loaded: {plugin.meta.name} (entry point)")

    graph = _ensure_db(config)
    graph.close()
    console.print(f"Database ready: {config.db_path}")
    _run_crg(CrgAdapter(project_root), project_root)


def execute_index(project_root: Path) -> None:
    from milknado.adapters.crg import CrgAdapter

    _load_or_default(project_root)
    _run_crg(CrgAdapter(project_root), project_root, rebuild=True)


def execute_status(project_root: Path, with_durations: bool) -> None:
    config = _load_or_default(project_root)
    graph = _ensure_db(config)
    try:
        print_status(graph, with_durations)
    finally:
        graph.close()


def execute_crg(project_root: Path) -> None:
    import json

    from milknado.adapters.crg import CrgAdapter

    if not (project_root / ".code-review-graph" / "graph.db").exists():
        console.print("[dim]No CRG index. Run [bold]milknado index[/bold] to build.[/dim]")
        raise typer.Exit(code=1)
    adapter = CrgAdapter(project_root)
    console.print(json.dumps(adapter.get_architecture_overview(), indent=2, default=str))


def execute_add_node(
    project_root: Path,
    description: str,
    parent: int | None,
    files: list[str] | None,
) -> None:
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


def execute_doctor(project_root: Path) -> None:
    from milknado.domains.common.doctor import render_report, run_doctor

    config_path = project_root / CONFIG_FILE
    config = _load_or_default(project_root)
    report = run_doctor(config_path, config)
    text, issue_count = render_report(report)
    typer.echo(text)
    if issue_count > 0:
        raise typer.Exit(code=1)


def execute_agents_check(project_root: Path) -> None:
    config = _load_or_default(project_root)
    check_agents_config(config, project_root)
