from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import MilknadoConfig, PluginHook, default_config, load_config
from milknado.domains.graph import MikadoGraph
from milknado.plugins import discover_entry_point_plugins, load_plugins


@dataclass(frozen=True)
class OpenProject:
    graph: MikadoGraph
    config: MilknadoConfig
    plugins: tuple[PluginHook, ...]


def load_project_config(root: Path) -> MilknadoConfig:
    root = root.resolve()
    config_path = root / "milknado.toml"
    return load_config(config_path) if config_path.exists() else default_config(root)


def load_project_plugins(config: MilknadoConfig) -> tuple[PluginHook, ...]:
    return (*load_plugins(config.plugins), *discover_entry_point_plugins())


def open_project_graph(
    config: MilknadoConfig,
    plugins: tuple[PluginHook, ...] = (),
) -> MikadoGraph:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    return MikadoGraph(config.db_path, plugins=plugins)


def open_project(root: Path, config: MilknadoConfig | None = None) -> OpenProject:
    config = config or load_project_config(root)
    plugins = load_project_plugins(config)
    return OpenProject(open_project_graph(config, plugins), config, plugins)
