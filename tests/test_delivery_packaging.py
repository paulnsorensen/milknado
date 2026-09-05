"""Hardening for the sliced-bread delivery crust (curd 1 of sliced-bread-crust).

The CLI/MCP surface is exposed ONLY via the ``milknado.cli`` / ``milknado.mcp``
packages — no flat ``cli_*.py`` / ``mcp_*.py`` modules at the package root — and
the console-script targets that back the ``milknado`` / ``milknado-mcp`` binaries
still resolve to callables. A regression that re-scatters delivery modules at the
root, or retargets a script to a module that no longer exports its entrypoint,
fails here.
"""

from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "milknado"
_PYPROJECT = _ROOT / "pyproject.toml"

_FLAT_DELIVERY_PREFIXES = ("cli_", "mcp_", "_cli_", "_mcp_")
_FLAT_DELIVERY_STEMS = {"_ralph_node_runner", "mcp_ralph"}


def test_no_flat_delivery_modules_at_package_root() -> None:
    offenders = sorted(
        path.name
        for path in _SRC.glob("*.py")
        if path.name.startswith(_FLAT_DELIVERY_PREFIXES) or path.stem in _FLAT_DELIVERY_STEMS
    )

    assert offenders == [], f"flat delivery modules leaked back to the package root: {offenders}"


def test_delivery_packages_exist() -> None:
    assert (_SRC / "cli" / "__init__.py").is_file()
    assert (_SRC / "mcp" / "__init__.py").is_file()


@pytest.mark.parametrize(
    ("target", "attr"),
    [("milknado.cli", "app"), ("milknado.mcp.server", "main")],
)
def test_console_script_targets_resolve(target: str, attr: str) -> None:
    module = import_module(target)

    entrypoint = getattr(module, attr)  # pyright: ignore[reportAny]
    assert callable(entrypoint), f"{target}:{attr} is not callable"


def test_pyproject_declares_packaged_script_targets() -> None:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]  # pyright: ignore[reportAny]

    assert scripts["milknado"] == "milknado.cli:app"
    assert scripts["milknado-mcp"] == "milknado.mcp.server:main"
