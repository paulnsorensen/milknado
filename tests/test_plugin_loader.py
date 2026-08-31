from __future__ import annotations

import importlib.metadata
import logging
from typing import NoReturn

import pytest

from milknado.plugins.loader import discover_entry_point_plugins, load_plugins


class _BrokenEntryPoint:
    name: str = "broken"

    def load(self) -> NoReturn:
        raise RuntimeError("constructor exploded")


def _raise_import_error(_name: str) -> NoReturn:
    raise ImportError("missing dependency")


def _broken_entry_points(*, group: str) -> list[_BrokenEntryPoint]:
    assert group == "milknado.plugins"
    return [_BrokenEntryPoint()]


def test_module_plugin_failure_logs_exception_context(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib, "import_module", _raise_import_error)
    with caplog.at_level(logging.ERROR, logger="milknado.plugins"):
        assert load_plugins(("broken.plugin",)) == []

    assert caplog.records[0].getMessage() == "Could not import plugin: broken.plugin"
    exc_info = caplog.records[0].exc_info
    assert exc_info is not None
    assert exc_info[1] is not None
    assert exc_info[1].args == ("missing dependency",)


def test_entry_point_failure_logs_exception_context(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", _broken_entry_points)
    with caplog.at_level(logging.ERROR, logger="milknado.plugins"):
        assert discover_entry_point_plugins() == []

    assert caplog.records[0].getMessage() == "Failed to load entry point plugin: broken"
    exc_info = caplog.records[0].exc_info
    assert exc_info is not None
    assert exc_info[1] is not None
    assert exc_info[1].args == ("constructor exploded",)
