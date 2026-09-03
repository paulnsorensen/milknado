"""Shared pytest fixtures for ralphify tests."""

import pytest
from _pytest.monkeypatch import MonkeyPatch

from milknado.loop.adapters import ADAPTERS


@pytest.fixture(autouse=True)
def _disable_streaming(monkeypatch: MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Force the blocking path on every registered adapter."""
    for adapter in ADAPTERS:
        monkeypatch.setattr(adapter, "supports_streaming", False)
