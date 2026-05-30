"""Tests for _cli_helpers.py shared CLI helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from milknado._cli_helpers import _ensure_plugins_loaded
from milknado.domains.common import PluginMeta


def _plugin(name: str) -> MagicMock:
    plugin = MagicMock()
    plugin.meta = PluginMeta(name=name, version="0.1.0", description="")
    return plugin


def test_ensure_plugins_loaded_collects_configured_and_entry_point_plugins() -> None:
    config = MagicMock()
    config.plugins = ("configured",)
    with (
        patch("milknado.plugins.load_plugins", return_value=[_plugin("configured")]),
        patch(
            "milknado.plugins.discover_entry_point_plugins",
            return_value=[_plugin("from_entry_point")],
        ),
        patch("milknado._cli_helpers.console", MagicMock()) as console,
    ):
        loaded = _ensure_plugins_loaded(config)

    assert [p.meta.name for p in loaded] == ["configured", "from_entry_point"]
    printed = "\n".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
    assert "Plugin loaded: configured" in printed
    assert "Plugin loaded: from_entry_point (entry point)" in printed
