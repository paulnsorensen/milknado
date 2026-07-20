"""Hardening for the import-linter contracts wired in curd 3 of sliced-bread-crust.

The layering rules (domains ↛ app/cli/mcp, app ↛ cli/mcp, adapters ↛ domains/app)
are both DECLARED in pyproject and currently HELD by the codebase. If a contract
is deleted or weakened the first test fails; if a forbidden import is introduced
the second fails — the same signal `just check-llm`'s import-contracts step gives,
pinned into the test suite so a regression cannot slip past a local run.
"""

from __future__ import annotations

from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# name -> (source_modules, forbidden_modules)
_EXPECTED_CONTRACTS: dict[str, tuple[list[str], list[str]]] = {
    "domains must not import app, cli, or mcp": (
        ["milknado.domains"],
        ["milknado.app", "milknado.cli", "milknado.mcp"],
    ),
    "app must not import cli or mcp": (
        ["milknado.app"],
        ["milknado.cli", "milknado.mcp"],
    ),
    "adapters must not import domains or app": (
        ["milknado.adapters"],
        ["milknado.domains", "milknado.app"],
    ),
}


def test_layering_contracts_are_declared() -> None:
    """Every layering contract is present in pyproject with its exact edges — so
    removing a contract or narrowing its forbidden set is caught here."""
    import importlinter.api as importlinter_api

    config = importlinter_api.read_configuration(config_filename=str(_PYPROJECT))
    declared = {
        contract["name"]: (
            list(contract["source_modules"]),
            list(contract["forbidden_modules"]),
        )
        for contract in config["contracts_options"]
        if contract["type"] == "forbidden"
    }

    for name, edges in _EXPECTED_CONTRACTS.items():
        assert name in declared, f"missing import-linter contract: {name!r}"
        assert declared[name] == edges, (
            f"contract {name!r} edges drifted from spec: {declared[name]!r}"
        )


def test_layering_contracts_hold() -> None:
    """The declared contracts are actually kept — introducing a forbidden import
    (e.g. domains importing app) turns this red, mirroring `lint-imports`."""
    from importlinter import configuration
    from importlinter.application import use_cases

    configuration.configure()
    kept = use_cases.lint_imports(config_filename=str(_PYPROJECT))

    assert kept is True, "import-linter reported a broken layering contract"
