"""Regression coverage for the sliced-bread import boundaries."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
_PYPROJECT = ROOT / "pyproject.toml"

_EXPECTED_CONTRACTS: dict[str, tuple[list[str], list[str]]] = {
    "domains must not import app, cli, or mcp": (
        ["milknado.domains"],
        ["milknado.app", "milknado.cli", "milknado.mcp"],
    ),
    "graph must not import execution": (
        ["milknado.domains.graph"],
        ["milknado.domains.execution"],
    ),
    "app must not import cli or mcp": (
        ["milknado.app"],
        ["milknado.cli", "milknado.mcp"],
    ),
    "execution controller must not import textual presentation": (
        ["milknado.app.run"],
        ["milknado.app.run_tui"],
    ),
    "adapters must not import application policy": (
        ["milknado.adapters"],
        ["milknado.app"],
    ),
}


def _contracts() -> dict[str, dict[str, Any]]:
    with _PYPROJECT.open("rb") as config_file:
        raw = tomllib.load(config_file)
    return {contract["name"]: contract for contract in raw["tool"]["importlinter"]["contracts"]}


def test_layering_contracts_are_declared() -> None:
    contracts = _contracts()
    for name, (source_modules, forbidden_modules) in _EXPECTED_CONTRACTS.items():
        assert contracts[name]["source_modules"] == source_modules
        assert contracts[name]["forbidden_modules"] == forbidden_modules

    barrel_contract = contracts["boundary modules must use domain barrels"]
    assert barrel_contract["source_modules"] == [
        "milknado.app",
        "milknado.cli",
        "milknado.mcp",
        "milknado.adapters",
    ]
    assert barrel_contract["forbidden_modules"] == [
        "milknado.domains.batching._model",
        "milknado.domains.batching._tarjan",
        "milknado.domains.dispatch._runstate",
        "milknado.domains.execution._context",
        "milknado.domains.execution.run_loop._completion",
        "milknado.domains.execution.run_loop._logging",
        "milknado.domains.execution.run_loop._result",
        "milknado.domains.github._fields",
        "milknado.domains.github._intent",
        "milknado.domains.graph._analytics_facade",
        "milknado.domains.graph._creation",
        "milknado.domains.graph._edge_facade",
        "milknado.domains.graph._goal_claims",
        "milknado.domains.graph._mutations",
        "milknado.domains.graph._persistence",
        "milknado.domains.graph._pipeline",
        "milknado.domains.graph._reads",
        "milknado.domains.graph._status",
        "milknado.domains.graph._transitions",
        "milknado.domains.wiki._locate",
        "milknado.domains.wiki._serialize",
    ]
    assert barrel_contract["allow_indirect_imports"] is True


def test_layering_contracts_hold() -> None:
    from importlinter import configuration
    from importlinter.application import use_cases

    configuration.configure()
    assert use_cases.lint_imports(config_filename=str(_PYPROJECT)) is True


def test_representative_boundary_modules_do_not_import_domain_submodules() -> None:
    source_root = ROOT / "src" / "milknado"
    domain_packages = (
        "batching",
        "common",
        "dispatch",
        "execution",
        "github",
        "graph",
        "planning",
        "reporting",
        "wiki",
    )
    for boundary in ("app", "cli", "mcp", "adapters"):
        for path in (source_root / boundary).rglob("*.py"):
            text = path.read_text()
            for package in domain_packages:
                assert f"from milknado.domains.{package}." not in text
                assert f"import milknado.domains.{package}." not in text


def test_application_policy_has_no_cli_presentation_dependencies() -> None:
    for module in ("plan.py", "run.py"):
        text = (ROOT / "src" / "milknado" / "app" / module).read_text()
        assert "import typer" not in text
        assert "from typer" not in text
        assert "from rich" not in text
        assert "Console(" not in text
