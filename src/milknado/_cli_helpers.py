"""Shared CLI helpers — config loading, DB init, plugin loading."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from milknado.domains.common import MilknadoConfig, default_config, load_config
from milknado.domains.graph import MikadoGraph

if TYPE_CHECKING:
    from milknado.domains.common.plugin import PluginHook

console = Console()

CONFIG_FILE = "milknado.toml"


def _find_config(project_root: Path) -> Path:
    return project_root / CONFIG_FILE


def _ensure_plugins_loaded(config: MilknadoConfig) -> list[PluginHook]:
    from milknado.plugins import discover_entry_point_plugins, load_plugins

    loaded: list[PluginHook] = []
    for plugin in load_plugins(config.plugins):
        console.print(f"  Plugin loaded: {plugin.meta.name}")
        loaded.append(plugin)
    for plugin in discover_entry_point_plugins():
        console.print(f"  Plugin loaded: {plugin.meta.name} (entry point)")
        loaded.append(plugin)
    return loaded


def _load_or_default(
    project_root: Path,
) -> tuple[MilknadoConfig, list[PluginHook]]:
    config_path = _find_config(project_root)
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = default_config(project_root)
    plugins = _ensure_plugins_loaded(config)
    return config, plugins


def _ensure_db(config: MilknadoConfig, plugins: list[PluginHook] | None = None) -> MikadoGraph:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    return MikadoGraph(config.db_path, plugins=plugins or ())


def _maybe_block_parent(graph: MikadoGraph, parent: int | None) -> None:
    if parent is None:
        return
    parent_node = graph.get_node(parent)
    if parent_node and parent_node.status.value == "running":
        graph.mark_blocked(parent)
        console.print(f"Parent node {parent} marked as blocked.")
