"""Shared CLI helpers — config loading, DB init, plugin loading."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from milknado.domains.common import MilknadoConfig
from milknado.domains.graph import MikadoGraph
from milknado.project import (
    OpenProject,
    load_project_config,
    load_project_plugins,
    open_project,
    open_project_graph,
)

if TYPE_CHECKING:
    from milknado.domains.common import PluginHook

console = Console()

CONFIG_FILE = "milknado.toml"


def _open_project(
    project_root: Path,
    config: MilknadoConfig | None = None,
) -> OpenProject:
    project = open_project(project_root, config)
    for plugin in project.plugins:
        console.print(f"  Plugin loaded: {plugin.meta.name}")
    return project


def _find_config(project_root: Path) -> Path:
    return project_root / CONFIG_FILE


def _ensure_plugins_loaded(config: MilknadoConfig) -> list[PluginHook]:
    plugins = list(load_project_plugins(config))
    for plugin in plugins:
        console.print(f"  Plugin loaded: {plugin.meta.name}")
    return plugins


def _load_or_default(
    project_root: Path,
) -> tuple[MilknadoConfig, list[PluginHook]]:
    config = load_project_config(project_root)
    plugins = _ensure_plugins_loaded(config)
    return config, plugins


def _ensure_db(config: MilknadoConfig, plugins: list[PluginHook] | None = None) -> MikadoGraph:
    return open_project_graph(config, tuple(plugins or ()))


def _maybe_block_parent(graph: MikadoGraph, parent: int | None) -> None:
    if parent is None:
        return
    parent_node = graph.get_node(parent)
    if parent_node and parent_node.status.value == "running":
        graph.mark_blocked(parent)
        console.print(f"Parent node {parent} marked as blocked.")


def _emit(text: str, out: Path | None) -> None:
    if out is None:
        typer.echo(text, nl=not text.endswith("\n"))
    else:
        out.write_text(text, encoding="utf-8")
