"""Shared CLI helpers — config loading, DB init, plugin loading."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
DEFAULT_PROJECT_ROOT = Path()

_TyperFactory = Callable[..., object]
_typer_option = cast(_TyperFactory, typer.Option)
_typer_argument = cast(_TyperFactory, typer.Argument)


def typer_option(*args: object, **kwargs: object) -> object:
    return _typer_option(*args, **kwargs)


def typer_argument(*args: object, **kwargs: object) -> object:
    return _typer_argument(*args, **kwargs)


def open_project_for_cli(
    project_root: Path,
    config: MilknadoConfig | None = None,
) -> OpenProject:
    project = open_project(project_root, config)
    for plugin in project.plugins:
        console.print(f"  Plugin loaded: {plugin.meta.name}")
    return project


def find_config(project_root: Path) -> Path:
    return project_root / CONFIG_FILE


def ensure_plugins_loaded(config: MilknadoConfig) -> list[PluginHook]:
    plugins = list(load_project_plugins(config))
    for plugin in plugins:
        console.print(f"  Plugin loaded: {plugin.meta.name}")
    return plugins


def load_or_default(
    project_root: Path,
) -> tuple[MilknadoConfig, list[PluginHook]]:
    config = load_project_config(project_root)
    plugins = ensure_plugins_loaded(config)
    return config, plugins


def ensure_db(config: MilknadoConfig, plugins: list[PluginHook] | None = None) -> MikadoGraph:
    return open_project_graph(config, tuple(plugins or ()))


def maybe_block_parent(graph: MikadoGraph, parent: int | None) -> None:
    if parent is None:
        return
    parent_node = graph.get_node(parent)
    if parent_node and parent_node.status.value == "running":
        graph.mark_blocked(parent)
        console.print(f"Parent node {parent} marked as blocked.")


def emit(text: str, out: Path | None) -> None:
    if out is None:
        typer.echo(text, nl=not text.endswith("\n"))
    else:
        _ = out.write_text(text, encoding="utf-8")


# Keep the historical private names available to callers that imported the CLI
# helpers directly; command modules use the public names above.
_open_project = open_project_for_cli
_find_config = find_config
_ensure_plugins_loaded = ensure_plugins_loaded
_load_or_default = load_or_default
_ensure_db = ensure_db
_maybe_block_parent = maybe_block_parent
_emit = emit

__all__ = [
    "DEFAULT_PROJECT_ROOT",
    "console",
    "emit",
    "ensure_db",
    "ensure_plugins_loaded",
    "find_config",
    "load_or_default",
    "maybe_block_parent",
    "open_project_for_cli",
    "typer_argument",
    "typer_option",
    "_emit",
    "_ensure_db",
    "_ensure_plugins_loaded",
    "_find_config",
    "_load_or_default",
    "_maybe_block_parent",
    "_open_project",
]
