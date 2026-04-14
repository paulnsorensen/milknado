from __future__ import annotations

import importlib
import logging
from typing import Any

from milknado.domains.common import PluginHook, PluginMeta

logger = logging.getLogger("milknado.plugins")


def _is_plugin(obj: Any) -> bool:
    return hasattr(obj, "meta") and hasattr(obj, "on_node_status_change")


def _find_plugin_class(module: Any) -> type | None:
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and _is_plugin(obj) and not name.startswith("_"):
            return obj
    return None


def load_plugins(plugin_names: tuple[str, ...]) -> list[PluginHook]:
    loaded: list[PluginHook] = []
    for name in plugin_names:
        try:
            module = importlib.import_module(name)
            cls = _find_plugin_class(module)
            if cls is None:
                logger.warning("No plugin class found in %s", name)
                continue
            instance = cls()
            meta: PluginMeta = instance.meta
            logger.info("Loaded plugin: %s v%s", meta.name, meta.version)
            loaded.append(instance)
        except ImportError:
            logger.warning("Could not import plugin: %s", name)
    return loaded


def discover_entry_point_plugins() -> list[PluginHook]:
    from importlib.metadata import entry_points

    loaded: list[PluginHook] = []
    for ep in entry_points(group="milknado.plugins"):
        try:
            cls = ep.load()
            instance = cls()
            meta: PluginMeta = instance.meta
            logger.info("Loaded plugin: %s v%s (entry point)", meta.name, meta.version)
            loaded.append(instance)
        except Exception:
            logger.warning("Failed to load entry point plugin: %s", ep.name)
    return loaded
