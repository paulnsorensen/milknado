from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from milknado.plugins.loader import discover_entry_point_plugins, load_plugins


def test_module_plugin_failure_logs_exception_context(caplog) -> None:
    with (
        patch(
            "milknado.plugins.loader.importlib.import_module",
            side_effect=ImportError("missing dependency"),
        ),
        caplog.at_level(logging.ERROR, logger="milknado.plugins"),
    ):
        assert load_plugins(("broken.plugin",)) == []

    assert caplog.records[0].getMessage() == "Could not import plugin: broken.plugin"
    assert caplog.records[0].exc_info is not None
    assert caplog.records[0].exc_info[1].args == ("missing dependency",)


def test_entry_point_failure_logs_exception_context(caplog) -> None:
    entry_point = MagicMock(name="broken")
    entry_point.name = "broken"
    entry_point.load.side_effect = RuntimeError("constructor exploded")
    with (
        patch("importlib.metadata.entry_points", return_value=[entry_point]),
        caplog.at_level(logging.ERROR, logger="milknado.plugins"),
    ):
        assert discover_entry_point_plugins() == []

    assert caplog.records[0].getMessage() == "Failed to load entry point plugin: broken"
    assert caplog.records[0].exc_info is not None
    assert caplog.records[0].exc_info[1].args == ("constructor exploded",)
